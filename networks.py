import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.utils.spectral_norm import spectral_norm
from torchvision.models import vgg16_bn
from torchvision.models.feature_extraction import create_feature_extractor

# ----------------------------------------------------------------------------------------------------------------------
#                                                     Common classes
# ----------------------------------------------------------------------------------------------------------------------
class BaseNetwork(nn.Module):
    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print("Network [{}] was created. Total number of parameters: {:.1f} million. "
              "To see the architecture, do print(network).".format(self.__class__.__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if 'BatchNorm2d' in classname:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif ('Conv' in classname or 'Linear' in classname) and hasattr(m, 'weight'):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError("initialization method '{}' is not implemented".format(init_type))
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, *inputs):
        pass

class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, args, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d,
                 use_sigmoid=False, num_D=2, getIntermFeat=False):
        super().__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(args, input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.init_weights(args.init_type, args.init_variance)

    def singleD_forward(self, model, inp):
        if self.getIntermFeat:
            result = [inp]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(inp)]

    def forward(self, inp):
        num_D = self.num_D
        result = []
        input_downsampled = inp
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Define the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    def __init__(self, args, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super().__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)
        
        self.init_weights(args.init_type, args.init_variance)

    def forward(self, inp):
        if self.getIntermFeat:
            res = [inp]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(inp)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True):
        super().__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, inp, target_is_real):
        if isinstance(inp[0], list):
            loss = 0
            for input_i in inp:
                pred = input_i[-1]
                target_tensor = torch.empty_like(pred).fill_(target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = torch.empty_like(inp[-1]).fill_(target_is_real)
            return self.loss(inp[-1], target_tensor)


# ----------------------------------------------------------------------------------------------------------------------
#                                              SegGenerator-related classes
# ----------------------------------------------------------------------------------------------------------------------
class SegGenerator(BaseNetwork):
    def __init__(self, args, input_nc, output_nc=13, norm_layer=nn.InstanceNorm2d):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU(),
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU(),
                                   nn.Conv2d(128, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU(),
                                   nn.Conv2d(256, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU())

        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1), norm_layer(1024), nn.ReLU(),
                                   nn.Conv2d(1024, 1024, kernel_size=3, padding=1), norm_layer(1024), nn.ReLU())

        self.up6 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv2d(1024, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU())

        self.up7 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv2d(512, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU(),
                                   nn.Conv2d(256, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU())

        self.up8 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv2d(256, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU(),
                                   nn.Conv2d(128, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU())

        self.up9 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv2d(128, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU())
        self.conv9 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU(),
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU(),
                                   nn.Conv2d(64, output_nc, kernel_size=3, padding=1))

        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.5)
        # self.sigmoid = nn.Sigmoid()

        self.init_weights(args.init_type, args.init_variance)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.drop(self.conv4(self.pool(conv3)))
        conv5 = self.drop(self.conv5(self.pool(conv4)))

        conv6 = self.conv6(torch.cat((conv4, self.up6(conv5)), 1))
        conv7 = self.conv7(torch.cat((conv3, self.up7(conv6)), 1))
        conv8 = self.conv8(torch.cat((conv2, self.up8(conv7)), 1))
        conv9 = self.conv9(torch.cat((conv1, self.up9(conv8)), 1))
        return conv9
        # return self.sigmoid(conv9)


# ----------------------------------------------------------------------------------------------------------------------
#                                                  GMM-related classes
# ----------------------------------------------------------------------------------------------------------------------
class FeatureExtraction(BaseNetwork):
    def __init__(self, input_nc, ngf=64, num_layers=4, norm_layer=nn.BatchNorm2d):
        super().__init__()

        nf = ngf
        layers = [nn.Conv2d(input_nc, nf, kernel_size=4, stride=2, padding=1), nn.ReLU(), norm_layer(nf)]

        for i in range(1, num_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1), nn.ReLU(), norm_layer(nf)]

        layers += [nn.Conv2d(nf, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(), norm_layer(512)]
        layers += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU()]

        self.model = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x):
        return self.model(x)


class FeatureCorrelation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, featureA, featureB):
        # Reshape features for matrix multiplication.
        b, c, h, w = featureA.size()
        featureA = featureA.permute(0, 3, 2, 1).reshape(b, w * h, c)
        featureB = featureB.reshape(b, c, h * w)

        # Perform matrix multiplication.
        corr = torch.bmm(featureA, featureB).reshape(b, w * h, h, w)
        return corr


class FeatureRegression(nn.Module):
    def __init__(self, input_nc=512, output_size=6, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, kernel_size=4, stride=2, padding=1), norm_layer(512), nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1), norm_layer(256), nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU()
        )
        self.linear = nn.Linear(64 * (input_nc // 16), output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x.reshape(x.size(0), -1))
        return self.tanh(x)

class BoundedGridLocNet(nn.Module):
    def __init__(self, args):
        super(BoundedGridLocNet, self).__init__()
        self.grid_size = args.grid_size
        self.rx, self.ry, self.cx, self.cy = torch.tensor(0.08, device='cuda'), torch.tensor(0.08, device='cuda'), torch.tensor(0.08, device='cuda'), torch.tensor(0.08, device='cuda')
        self.rg, self.cg = torch.tensor(0.02, device='cuda'), torch.tensor(0.02, device='cuda')

    def forward(self, coor):
        # coor [batch_size, -1, 2]
        row = self.get_row(coor, self.grid_size)
        col = self.get_col(coor, self.grid_size)
        rg_loss = sum(self.grad_row(coor, self.grid_size))
        cg_loss = sum(self.grad_col(coor, self.grid_size))
        rg_loss = torch.max(rg_loss, self.rg)
        cg_loss = torch.max(cg_loss, self.cg)
        row_x,row_y=row[:,:,0], row[:,:,1]
        col_x,col_y=col[:,:,0], col[:,:,1]
        rx_loss=torch.max(self.rx, row_x).mean()
        ry_loss=torch.max(self.ry, row_y).mean()
        cx_loss=torch.max(self.cx, col_x).mean()
        cy_loss=torch.max(self.cy, col_y).mean()
        return rx_loss, ry_loss, cx_loss, cy_loss, rg_loss, cg_loss

    def get_row(self, coor, num):
        sec_dic=[]
        for j in range(num):
            sum=0
            buffer=0
            flag=False
            for i in range(num-1):
                differ=(coor[:,j*num+i+1,:]-coor[:,j*num+i,:])**2
                if not flag:
                    second_dif=0
                    flag=True
                else:
                    second_dif=torch.abs(differ-buffer)
                    sec_dic.append(second_dif)
                buffer=differ
                sum+=second_dif
        return torch.stack(sec_dic,dim=1)

    def get_col(self, coor, num):
        sec_dic=[]
        for i in range(num):
            sum = 0
            buffer = 0
            flag = False
            for j in range(num - 1):
                differ = (coor[:, (j+1) * num + i , :] - coor[:, j * num + i, :]) ** 2
                if not flag:
                    second_dif = 0
                    flag = True
                else:
                    second_dif = torch.abs(differ-buffer)
                    sec_dic.append(second_dif)
                buffer = differ
                sum += second_dif
        return torch.stack(sec_dic,dim=1)

    def grad_row(self, coor, num):
        sec_term = []
        for j in range(num):
            for i in range(1, num - 1):
                x0, y0 = coor[:, j * num + i - 1, :][0]
                x1, y1 = coor[:, j * num + i + 0, :][0]
                x2, y2 = coor[:, j * num + i + 1, :][0]
                grad = torch.abs((y1 - y0) * (x1 - x2) - (y1 - y2) * (x1 - x0))
                sec_term.append(grad)
        return sec_term

    def grad_col(self, coor, num):
        sec_term = []
        for i in range(num):
            for j in range(1, num - 1):
                x0, y0 = coor[:, (j - 1) * num + i, :][0]
                x1, y1 = coor[:, j * num + i, :][0]
                x2, y2 = coor[:, (j + 1) * num + i, :][0]
                grad = torch.abs((y1 - y0) * (x1 - x2) - (y1 - y2) * (x1 - x0))
                sec_term.append(grad)
        return sec_term


class TpsGridGen(nn.Module):
    def __init__(self, args, dtype=torch.float):
        super().__init__()
        # Create a grid in numpy.
        # TODO: set an appropriate interval ([-1, 1] in CP-VTON, [-0.9, 0.9] in the current version of VITON-HD)
        grid_X, grid_Y = np.meshgrid(np.linspace(-0.9, 0.9, args.load_width), np.linspace(-0.9, 0.9, args.load_height))
        self.grid_X = torch.tensor(grid_X, dtype=dtype, device='cuda').unsqueeze(0).unsqueeze(3)  # size: (1, h, w, 1)
        self.grid_Y = torch.tensor(grid_Y, dtype=dtype, device='cuda').unsqueeze(0).unsqueeze(3)  # size: (1, h, w, 1)

        # Initialize the regular grid for control points P.
        self.N = args.grid_size * args.grid_size
        coords = np.linspace(-0.9, 0.9, args.grid_size)
        # FIXME: why P_Y and P_X are swapped?
        P_Y, P_X = np.meshgrid(coords, coords)
        P_X = torch.tensor(P_X, dtype=dtype, device='cuda').reshape(self.N, 1)
        P_Y = torch.tensor(P_Y, dtype=dtype, device='cuda').reshape(self.N, 1)
        self.P_X_base = P_X.clone()
        self.P_Y_base = P_Y.clone()

        self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
        self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)  # size: (1, 1, 1, 1, self.N)
        self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)  # size: (1, 1, 1, 1, self.N)

    # TODO: refactor
    def compute_L_inverse(self,X,Y):
        N = X.size(0) # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N,N)
        Ymat = Y.expand(N,N)
        P_dist_squared = torch.pow(Xmat-Xmat.transpose(0,1),2)+torch.pow(Ymat-Ymat.transpose(0,1),2)
        P_dist_squared[P_dist_squared==0]=1 # make diagonal 1 to avoid NaN in log computation
        K = torch.mul(P_dist_squared,torch.log(P_dist_squared))
        # construct matrix L
        O = torch.ones(N,1, device='cuda')
        Z = torch.zeros(3,3, device='cuda')
        P = torch.cat((O,X,Y),1)
        L = torch.cat((torch.cat((K,P),1), torch.cat((P.transpose(0,1),Z),1)),0)
        Li = torch.inverse(L)
        return Li

    # TODO: refactor
    def forward(self, theta):
        points = torch.cat((self.grid_X, self.grid_Y), 3)
        batch_size = theta.size(0)

        # inp are the corresponding control points P_i
        # split theta into point coordinates
        Q_X = theta[:,:,0:1]
        Q_Y = theta[:,:,1:2]
        # Q_X=theta[:,:self.N,:,:].squeeze(3)
        # Q_Y=theta[:,self.N:,:,:].squeeze(3)
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1,points_h,points_w,1,self.N))
        P_Y = self.P_Y.expand((1,points_h,points_w,1,self.N))

        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_X)
        W_Y = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_X)
        A_Y = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:,:,:,0].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,0].size()+(1,self.N))
        points_Y_for_summation = points[:,:,:,1].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,1].size()+(1,self.N))

        if points_b==1:
            delta_X = points_X_for_summation-P_X
            delta_Y = points_Y_for_summation-P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation-P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation-P_Y.expand_as(points_Y_for_summation)

        dist_squared = torch.pow(delta_X,2)+torch.pow(delta_Y,2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared==0]=1 # avoid NaN in log computation
        U = torch.mul(dist_squared,torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:,:,:,0].unsqueeze(3)
        points_Y_batch = points[:,:,:,1].unsqueeze(3)
        if points_b==1:
            points_X_batch = points_X_batch.expand((batch_size,)+points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,)+points_Y_batch.size()[1:])

        points_X_prime = A_X[:,:,:,:,0]+ \
                       torch.mul(A_X[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_X[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_X,U.expand_as(W_X)),4)

        points_Y_prime = A_Y[:,:,:,:,0]+ \
                       torch.mul(A_Y[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_Y[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_Y,U.expand_as(W_Y)),4)

        return torch.cat((points_X_prime, points_Y_prime), 3)


class GMM(nn.Module):
    def __init__(self, args, inputA_nc, inputB_nc):
        super().__init__()
        self.extractionA = FeatureExtraction(inputA_nc, ngf=64, num_layers=4)
        self.extractionB = FeatureExtraction(inputB_nc, ngf=64, num_layers=4)
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(input_nc=(args.load_width // 64) * (args.load_height // 64),
                                            output_size=2 * args.grid_size**2)
        self.gridGen = TpsGridGen(args)

    def forward(self, inputA, inputB):
        featureA = F.normalize(self.extractionA(inputA), dim=1)
        featureB = F.normalize(self.extractionB(inputB), dim=1)
        corr = self.correlation(featureA, featureB)
        theta = self.regression(corr)
        batch_size = theta.size(0)
        theta = theta.view(batch_size, -1, 2)
        warped_grid = self.gridGen(theta)
        return theta, warped_grid


# ----------------------------------------------------------------------------------------------------------------------
#                                             ALIASGenerator-related classes
# ----------------------------------------------------------------------------------------------------------------------
class MaskNorm(nn.Module):
    def __init__(self, norm_nc):
        super().__init__()

        self.norm_layer = nn.InstanceNorm2d(norm_nc, affine=False)

    def normalize_region(self, region, mask):
        b, c, h, w = region.size()

        num_pixels = mask.sum((2, 3), keepdim=True)  # size: (b, 1, 1, 1)
        num_pixels[num_pixels == 0] = 1
        mu = region.sum((2, 3), keepdim=True) / num_pixels  # size: (b, c, 1, 1)

        normalized_region = self.norm_layer(region + (1 - mask) * mu)
        return normalized_region * torch.sqrt(num_pixels / (h * w))

    def forward(self, x, mask):
        mask = mask.detach()
        normalized_foreground = self.normalize_region(x * mask, mask)
        normalized_background = self.normalize_region(x * (1 - mask), 1 - mask)
        return normalized_foreground + normalized_background


class ALIASNorm(nn.Module):
    def __init__(self, norm_type, norm_nc, label_nc):
        super().__init__()

        self.noise_scale = nn.Parameter(torch.zeros(norm_nc))

        assert norm_type.startswith('alias')
        param_free_norm_type = norm_type[len('alias'):]
        if param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'mask':
            self.param_free_norm = MaskNorm(norm_nc)
        else:
            raise ValueError(
                "'{}' is not a recognized parameter-free normalization type in ALIASNorm".format(param_free_norm_type)
            )

        nhidden = 128
        ks = 3
        pw = ks // 2
        self.conv_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.conv_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.conv_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, seg, misalign_mask=None):
        # Part 1. Generate parameter-free normalized activations.
        b, c, h, w = x.size()
        noise = (torch.randn(b, w, h, 1, device='cuda') * self.noise_scale).transpose(1, 3)

        if misalign_mask is None:
            normalized = self.param_free_norm(x + noise)
        else:
            normalized = self.param_free_norm(x + noise, misalign_mask)

        # Part 2. Produce affine parameters conditioned on the segmentation map.
        actv = self.conv_shared(seg)
        gamma = self.conv_gamma(actv)
        beta = self.conv_beta(actv)

        # Apply the affine parameters.
        output = normalized * (1 + gamma) + beta
        return output


class ALIASResBlock(nn.Module):
    def __init__(self, args, input_nc, output_nc, use_mask_norm=True):
        super().__init__()

        self.learned_shortcut = (input_nc != output_nc)
        middle_nc = min(input_nc, output_nc)

        self.conv_0 = nn.Conv2d(input_nc, middle_nc, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(middle_nc, output_nc, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(input_nc, output_nc, kernel_size=1, bias=False)

        subnorm_type = args.norm_G
        if subnorm_type.startswith('spectral'):
            subnorm_type = subnorm_type[len('spectral'):]
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        semantic_nc = args.semantic_nc
        if use_mask_norm:
            subnorm_type = 'aliasmask'
            semantic_nc = semantic_nc + 1

        self.norm_0 = ALIASNorm(subnorm_type, input_nc, semantic_nc)
        self.norm_1 = ALIASNorm(subnorm_type, middle_nc, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = ALIASNorm(subnorm_type, input_nc, semantic_nc)

        self.relu = nn.LeakyReLU(0.2)

    def shortcut(self, x, seg, misalign_mask):
        if self.learned_shortcut:
            return self.conv_s(self.norm_s(x, seg, misalign_mask))
        else:
            return x

    def forward(self, x, seg, misalign_mask=None):
        seg = F.interpolate(seg, size=x.size()[2:], mode='nearest')
        if misalign_mask is not None:
            misalign_mask = F.interpolate(misalign_mask, size=x.size()[2:], mode='nearest')

        x_s = self.shortcut(x, seg, misalign_mask)

        dx = self.conv_0(self.relu(self.norm_0(x, seg, misalign_mask)))
        dx = self.conv_1(self.relu(self.norm_1(dx, seg, misalign_mask)))
        output = x_s + dx
        return output


class ALIASGenerator(BaseNetwork):
    def __init__(self, args, input_nc):
        super().__init__()
        self.num_upsampling_layers = args.num_upsampling_layers

        self.sh, self.sw = self.compute_latent_vector_size(args)

        nf = args.ngf
        self.conv_0 = nn.Conv2d(input_nc, nf * 16, kernel_size=3, padding=1)
        for i in range(1, 8):
            self.add_module('conv_{}'.format(i), nn.Conv2d(input_nc, 16, kernel_size=3, padding=1))

        self.head_0 = ALIASResBlock(args, nf * 16, nf * 16)

        self.G_middle_0 = ALIASResBlock(args, nf * 16 + 16, nf * 16)
        self.G_middle_1 = ALIASResBlock(args, nf * 16 + 16, nf * 16)

        self.up_0 = ALIASResBlock(args, nf * 16 + 16, nf * 8)
        self.up_1 = ALIASResBlock(args, nf * 8 + 16, nf * 4)
        self.up_2 = ALIASResBlock(args, nf * 4 + 16, nf * 2, use_mask_norm=False)
        self.up_3 = ALIASResBlock(args, nf * 2 + 16, nf * 1, use_mask_norm=False)
        if self.num_upsampling_layers == 'most':
            self.up_4 = ALIASResBlock(args, nf * 1 + 16, nf // 2, use_mask_norm=False)
            nf = nf // 2

        self.conv_img = nn.Conv2d(nf, 3, kernel_size=3, padding=1)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

        self.init_weights(args.init_type, args.init_variance)

    def compute_latent_vector_size(self, args):
        if self.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif self.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif self.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError(f"args.num_upsampling_layers '{self.num_upsampling_layers}' is not recognized")

        sh = args.load_height // 2**num_up_layers
        sw = args.load_width // 2**num_up_layers
        return sh, sw

    def forward(self, x, seg, seg_div, misalign_mask):
        samples = [F.interpolate(x, size=(self.sh * 2**i, self.sw * 2**i), mode='nearest') for i in range(8)]
        features = [self._modules['conv_{}'.format(i)](samples[i]) for i in range(8)]

        x = self.head_0(features[0], seg_div, misalign_mask)

        x = self.up(x)
        x = self.G_middle_0(torch.cat((x, features[1]), 1), seg_div, misalign_mask)
        if self.num_upsampling_layers in ['more', 'most']:
            x = self.up(x)
        x = self.G_middle_1(torch.cat((x, features[2]), 1), seg_div, misalign_mask)

        x = self.up(x)
        x = self.up_0(torch.cat((x, features[3]), 1), seg_div, misalign_mask)
        x = self.up(x)
        x = self.up_1(torch.cat((x, features[4]), 1), seg_div, misalign_mask)
        x = self.up(x)
        x = self.up_2(torch.cat((x, features[5]), 1), seg)
        x = self.up(x)
        x = self.up_3(torch.cat((x, features[6]), 1), seg)
        if self.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(torch.cat((x, features[7]), 1), seg)

        x = self.conv_img(self.relu(x))
        return self.tanh(x)


def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1, 2))/(c*h*w)


class VGGLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer_ids = [22, 32, 42]
        self.weights = [5, 15, 4]
        m = vgg16_bn(pretrained=True).features.eval()
        return_nodes = {f'{x}': f'feat{i}' for i, x in enumerate(layer_ids)}
        self.vgg_fx = create_feature_extractor(m, return_nodes=return_nodes)
        self.vgg_fx.requires_grad_(False)
        self.l1_loss = nn.L1Loss()

    def forward(self, x, y):
        x_vgg = self.vgg_fx(x)
        with torch.no_grad():
            y_vgg = self.vgg_fx(y)
        loss = self.l1_loss(x, y)
        for i, k in enumerate(x_vgg.keys()):
            loss += self.weights[i] * self.l1_loss(x_vgg[k], y_vgg[k].detach_())       # feature loss
            loss += self.weights[i]**2 * 5e3 * self.l1_loss(gram_matrix(x_vgg[k]), gram_matrix(y_vgg[k]))  # style loss
        return loss