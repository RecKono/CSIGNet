import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.Res2Net_v1b import res2net101_v1b_26w_4s
from lib.pvt2 import pvt_v2_b5
############change is best-----------------------
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,isbn=True,isrelu=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)

        self.bn = nn.BatchNorm2d(out_planes) if isbn else None
        self.relu = nn.ReLU(inplace=True)   if isrelu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x=self.relu(x)
        return x
class SEA(nn.Module):
    def __init__(self):
        super(SEA, self).__init__()
        self.conv1h = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2h = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3h = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4h = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv1v = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2v = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3v = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4v = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv_cat = BasicConv2d(64+64, 64, kernel_size=3, stride=1, padding=1)
        self.weight=nn.Parameter(torch.ones(2))
        
        self.conv5 = BasicConv2d(64+64, 64, kernel_size=3, stride=1, padding=1,isrelu=False)
        self.conv_out = nn.Conv2d(64, 1, 1)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        out1h = self.conv1h(left)
        out2h = self.conv2h(out1h)
        out1v = self.conv1v(down)
        out2v = self.conv2v(out1v)

        fuse1 = out2h*out2v
        fuse2 = self.conv_cat(torch.cat((out2h,out2v),1))
        fuse  = self.weight[0]*fuse1+self.weight[1]*fuse2

        out3h = self.conv3h(fuse)+out1h
        out4h = self.conv4h(out3h)
        out3v = self.conv3v(fuse)+out1v
        out4v = self.conv4v(out3v)

        edge_feature = self.conv5(torch.cat((out4h, out4v), dim=1))
        edge_out = self.conv_out(edge_feature)

        return edge_feature, edge_out
class Fusion(nn.Module):
    def __init__(self,channel):
        super(Fusion, self).__init__()
        self.conv1=nn.ModuleList([
            BasicConv2d(channel,channel,3,padding=1) for i in range(3)
        ])
        self.conv2 = nn.ModuleList([
            BasicConv2d(channel, channel, 3, padding=1) for i in range(3)
        ])
        self.conv3 = nn.ModuleList([
            BasicConv2d(channel, channel, 3, padding=1) for i in range(3)
        ])
        self.conv4 = nn.ModuleList([
            BasicConv2d(channel, channel, 3, padding=1) for i in range(3)
        ])
        self.conv5 = nn.ModuleList([
            BasicConv2d(channel, channel, 3, padding=1) for i in range(3)
        ])

    def forward(self,x1,_x2=None,_x3=None):
        outs=[]
        if _x2 is not None:
            x2=_x2.copy(); x3=_x3.copy()
        for i in range(3):
            if _x2 is None:
                outs.append(self.conv1[i](x1[i]))
            else:
                add=[x1[0],
                     F.interpolate(x2[1],size=x1[i].size()[2:],mode='bilinear'),
                     F.interpolate(x3[2],size=x1[i].size()[2:],mode='bilinear')]
                if x2[i].size()[2:]!=x1[i].size()[2:]:
                    x2[i]=F.interpolate(x2[i],size=x1[i].size()[2:],mode='bilinear')
                if x3[i].size()[2:]!=x1[i].size()[2:]:
                    x3[i]=F.interpolate(x3[i],size=x1[i].size()[2:],mode='bilinear')
                f1=self.conv1[i](x1[i])
                f2=self.conv2[i](x2[i])
                f3=self.conv3[i](x3[i])
                fuse=self.conv4[i](f1*f2*f3)
                fuse=fuse+self.conv5[i](add[i])
                outs.append(fuse)

        return outs
class Mshape(nn.Module):
    def __init__(self,channel):
        super(Mshape, self).__init__()
        self.fusion1=Fusion(channel=channel)
        self.fusion2=Fusion(channel=channel)
        self.fusion3=Fusion(channel=channel)
        self.fusion4=Fusion(channel=channel)
        self.fusion5=Fusion(channel=channel)
        self.conv=nn.ModuleList([
            BasicConv2d(2*channel,channel,3,padding=1) for i in range(3)
        ])

        self.conv_cat1=BasicConv2d(3*channel,channel,3,padding=1)
        self.conv_cat2=BasicConv2d(3*channel,channel,3,padding=1)
        self.conv_cat3=BasicConv2d(3*channel,channel,3,padding=1)
        self.conv_cat4=BasicConv2d(3*channel,channel,3,padding=1)
        self.conv_cat5=BasicConv2d(3*channel,channel,3,padding=1)

    def forward(self,x2,x3,x4,x5):
        # x1=[x1,x1,x1]
        x2=[x2,x2,x2]
        x3=[x3,x3,x3]
        x4=[x4,x4,x4]
        x5=[x5,x5,x5]
        f5=self.fusion5(x5)
        f4=self.fusion4(x4,x3,f5)
        f3=self.fusion3(x3,f4,f5)
        f2=self.fusion2(x2,f3,f4)

        outs=[]
        for i in range(3):
            x=torch.cat((f4[i],F.interpolate(f5[i],scale_factor=2,mode='bilinear')),1)
            outs.append(self.conv[i](x))
        f1=self.fusion1(f2,f3,outs)

        out5=self.conv_cat5(torch.cat((f5[0],f5[1],f5[2]),1))
        out4=self.conv_cat4(torch.cat((f4[0],f4[1],f4[2]),1))
        out3=self.conv_cat3(torch.cat((f3[0],f3[1],f3[2]),1))
        out2=self.conv_cat2(torch.cat((f2[0],f2[1],f2[2]),1))
        out1=self.conv_cat1(torch.cat((f1[0],f1[1],f1[2]),1))

        return out1,out2,out3,out4,out5


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class UCSC(nn.Module):
    def __init__(self,channel,size):
        super(UCSC, self).__init__()
        self.conv1=BasicConv2d(channel,channel,3,padding=1)
        self.weights=nn.Parameter(torch.rand(1,size*2,size*2))
        self.biass=nn.Parameter(torch.rand(1,size*2,size*2))
        self.conv2=BasicConv2d(channel,channel,3,padding=1)
        self.conv3=BasicConv2d(channel,channel,3,padding=1)
        self.conv4=BasicConv2d(channel,channel,3,padding=1)
        self.conv5=BasicConv2d(channel,channel,3,padding=1)



    def forward(self,x):
        x=F.interpolate(x,scale_factor=2,mode='bilinear')
        x=self.conv1(x)

        cweight=F.adaptive_avg_pool2d(x,(1,1))+F.adaptive_max_pool2d(x,(1,1))
        xc=cweight*x
        xc=self.conv2(xc)
        xs=self.weights*x+self.biass
        xs=self.conv3(xs)
        fuse=xc*xs
        x=self.conv4(fuse)+x
        x=self.conv5(x)
        return x
class NCD1(nn.Module):
    def __init__(self,channel):
        super(NCD1, self).__init__()
        self.u1=UCSC(channel,11)
        self.u2=UCSC(channel,22)
        self.u3=UCSC(channel*2,22)

        self.conv1=BasicConv2d(channel,channel,3,padding=1)
        self.conv2=BasicConv2d(channel,channel,3,padding=1)
        self.conv3=BasicConv2d(channel,channel,3,padding=1)
        self.conv4=BasicConv2d(channel,channel,3,padding=1)
        self.conv5=BasicConv2d(channel,channel,3,padding=1)
        self.conv6=BasicConv2d(3*channel,channel,3,padding=1)
        self.conv_cat=BasicConv2d(2*channel,channel,3,padding=1)
        self.conv_res=BasicConv2d(channel,1,1,isrelu=False)


    def forward(self,x2,x3,x4):
        x4=self.u1(self.conv1(x4))
        x3=self.conv2(x3)
        x2=self.conv3(x2)

        x3_1=x3+self.conv4(x3*x4)
        x3_1=self.u2(x3_1)
        x3_2=torch.cat((x3,self.conv4(x3*x4)),1)
        x3_2=self.u3(x3_2)

        x2_1=x2+self.conv5(x2*x3_1)
        x2_2=torch.cat((x3_2,self.conv5(x2*x3_1)),1)
        x2_2=self.conv6(x2_2)
        fea=self.conv_cat(torch.cat((x2_1,x2_2),1))
        pre=self.conv_res(fea)

        return fea,pre
class HMU(nn.Module):
    def __init__(self,channel, subchannel):
        super(HMU, self).__init__()
        self.group = channel // subchannel
        self.hidden_dim=(int)(channel/self.group)
        self.interact = nn.ModuleDict()
        self.interact["0"]=BasicConv2d(self.hidden_dim+1,3*self.hidden_dim,3,padding=1)
        for group_id in range(1,self.group-1):
            self.interact[str(group_id)]=BasicConv2d(2*self.hidden_dim+2,3*self.hidden_dim,3,padding=1)
        self.interact[str(self.group-1)]=BasicConv2d(2*self.hidden_dim+2,2*self.hidden_dim,3,padding=1)
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channel, channel, 1),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, 1),
            nn.Softmax(dim=1),
        )
        self.conv=BasicConv2d(channel,1,3,padding=1,isrelu=False)

    def forward(self,x,y):
        if self.group == 1:
            xs = x
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
        elif self.group == 16:
            xs = torch.chunk(x, 16, dim=1)
        elif self.group == 32:
            xs = torch.chunk(x, 32, dim=1)
        elif self.group == 64:
            xs = torch.chunk(x, 64, dim=1)
        else:
            raise Exception("Invalid Channel")

        outs=[]

        branch_out=self.interact["0"](torch.cat((xs[0],y),1))
        outs.append(branch_out.chunk(3,dim=1))
        for group_id in range(1,self.group-1):
            x_=torch.cat((xs[group_id],y,outs[group_id-1][1],y),1)
            branch_out=self.interact[str(group_id)](x_)
            outs.append(branch_out.chunk(3,dim=1))
        x_=torch.cat((xs[self.group-1],y,outs[self.group-2][0],y),1)
        branch_out=self.interact[str(self.group-1)](x_)
        outs.append(branch_out.chunk(2,dim=1))

        g2=torch.cat([i[0] for i in outs],dim=1)
        g3=torch.cat([i[-1] for i in outs],dim=1)
        g2=self.gate_genator(g2)
        x=g2*g3+x
        y=self.conv(x)
        return x,y
class ReverseStage(nn.Module):
    def __init__(self, channel,size):
        super(ReverseStage, self).__init__()
        self.weak_gra = HMU(channel, 16)
        self.medium_gra = HMU(channel, 8)
        self.strong_gra = HMU(channel, 4)


        self.w=nn.Parameter(torch.rand(size,size))
        self.b=nn.Parameter(torch.rand(size,size))
        self.conv_c1=BasicConv2d(1,channel,3,padding=1)
        self.conv_c2=BasicConv2d(channel,1,3,padding=1)
        self.conv=BasicConv2d(1,1,1,isrelu=False)

    def forward(self, x, y,edge):
        # reverse guided block
        y = -1 * (torch.sigmoid(y)) + 1

        # three group-reversal attention blocks
        x, y = self.weak_gra(x, y)
        x, y = self.medium_gra(x, y)
        _, y = self.strong_gra(x, y)


        edge=F.interpolate(edge,size=y.size()[2:],mode='bilinear')
        edge=self.conv_c2(self.conv_c1(edge))

        y=y*self.w*edge+self.b*edge
        

        return y


















class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(Network, self).__init__()
        # ---- ResNet Backbone ----
        self.backbone = pvt_v2_b5()  # [64, 128, 320, 512]
        path = '/home/rec/Desktop/SInet/SINet-V2-main/snapshot/pvtv2/pvt_v2_b5.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        # ---- Receptive Field Block like module ----


        self.rfb1_1 = RFB_modified(64, channel)
        self.rfb2_1 = RFB_modified(128, channel)
        self.rfb3_1 = RFB_modified(320, channel)
        self.rfb4_1 = RFB_modified(512, channel)
        # ---- Partial Decoder ----
        self.sea=SEA()
        self.mshape=Mshape(channel=channel)
        self.NCD = NCD1(channel)

        # # ---- reverse stage ----
        self.RS5 = ReverseStage(channel,11)
        self.RS4 = ReverseStage(channel,22)
        self.RS3 = ReverseStage(channel,44)



    def forward(self, x):
        # Feature Extraction
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        
        # Receptive Field Block (enhanced)
        x1_rfb = self.rfb1_1(x1)
        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)        # channel -> 32
        x,x1,x2_rfb,x3_rfb,x4_rfb=self.mshape(x1_rfb,x2_rfb,x3_rfb,x4_rfb)
        
        
        _,edge_p=self.sea(x,x1)
        # Neighbourhood Connected Decoder
        _,S_g = self.NCD(x2_rfb, x3_rfb, x4_rfb)
        S_g_pred = F.interpolate(S_g, scale_factor=8, mode='bilinear')    # Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- reverse stage 5 ----
        guidance_g = F.interpolate(S_g, scale_factor=0.25, mode='bilinear')
        ra4_feat = self.RS5(x4_rfb, guidance_g,edge_p)
        S_5 = ra4_feat + guidance_g+F.interpolate(edge_p,scale_factor=0.125,mode='bilinear')
        S_5_pred = F.interpolate(S_5, scale_factor=32, mode='bilinear')  # Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse stage 4 ----
        guidance_5 = F.interpolate(S_5, scale_factor=2, mode='bilinear')
        ra3_feat = self.RS4(x3_rfb, guidance_5,edge_p)
        S_4 = ra3_feat + guidance_5+F.interpolate(edge_p,scale_factor=0.25,mode='bilinear')
        S_4_pred = F.interpolate(S_4, scale_factor=16, mode='bilinear')  # Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse stage 3 ----
        guidance_4 = F.interpolate(S_4, scale_factor=2, mode='bilinear')
        ra2_feat = self.RS3(x2_rfb, guidance_4,edge_p)
        S_3 = ra2_feat + guidance_4+F.interpolate(edge_p,scale_factor=0.5,mode='bilinear')
        S_3_pred = F.interpolate(S_3, scale_factor=8, mode='bilinear')   # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return S_g_pred, S_5_pred, S_4_pred, S_3_pred,edge_p
        # F.interpolate(edge_p,scale_factor=4,mode='bilinear')


if __name__ == '__main__':
    import numpy as np
    from time import time
    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 352, 352)
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate=float(end-start)
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
