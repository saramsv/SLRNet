import torch
import torch.nn as nn
import numpy as np
from . import resnet, resnext, mobilenet, hrnet, xception, xception65
from semseg.lib.nn import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d
import cv2

# Helper functions 
def get_activation(name, activation):
    def hook(model,input, output):
        try: 
            activation[name] = output.detach()
        except:
            activation[name] = []
            for out in output:
                activation[name].append(out.detach())
    return hook

def register_hooks(model, module_names, activation, show=False):
    for name, module in model.named_children():
        if name in module_names:
            module.register_forward_hook(get_activation('{}'.format(name), activation))
            if show: print("  -register_hooks: {}".format(name))

# end

class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, batch_size, training_type=None, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale
        self.batch_size = batch_size
        self.training_type = training_type


    def forward(self, feed_dict, *, epoch_weight=1, weight_type='eta', segSize=None):
        sup_feed_dict, seq_feed_dict, sup_pred, seq_pred = None, None, None, None

        if isinstance(feed_dict, tuple):
            sup_feed_dict = feed_dict[0]
            seq_feed_dict = feed_dict[1] # get the data for the second iterator that has the seq info in it
            if type(sup_feed_dict) is list and type(seq_feed_dict) is list:
                if torch.cuda.is_available():
                    sup_feed_dict = sup_feed_dict[0]
                    seq_feed_dict = seq_feed_dict[0] # Single valued list
                    sup_feed_dict['img_data'] = sup_feed_dict['img_data'].cuda()
                    sup_feed_dict['seg_label'] = sup_feed_dict['seg_label'].cuda()
                    seq_feed_dict['img_data'] = seq_feed_dict['img_data'].cuda()
                    seq_feed_dict['seg_label'] = seq_feed_dict['seg_label'].cuda()
                else:
                    raise RunTimeError('Cannot convert torch.Floattensor into torch.cuda.FloatTensor')
        else:
            sup_feed_dict = feed_dict
            # added the following for training on 1 gpu
            # if the dataset is loaded as a list, this will
            # raise a TypeError while trying to access it as a dictionary.
            if type(sup_feed_dict) is list:
                sup_feed_dict = sup_feed_dict[0]
                # also, convert to torch.cuda.FloatTensor
                if torch.cuda.is_available():
                    sup_feed_dict['img_data'] = sup_feed_dict['img_data'].cuda()
                    sup_feed_dict['seg_label'] = sup_feed_dict['seg_label'].cuda()
                else:
                    raise RunTimeError('Cannot convert torch.Floattensor into torch.cuda.FloatTensor')
        
        # training
		#### start for hidden layers
        activation = {}
        
        '''
        # def get_activation(name):
        def get_activation(name, activation):
            def hook(model,input, output):
                activation[name] = output.detach()
            return hook
        '''

        # self.decoder.cbr.register_forward_hook(get_activation('cbr'))
        # self.decoder.conv_last.register_forward_hook(get_activation('conv_last'))
        hidden_wt = weight_type.split(',')
        if len(list(set(hidden_wt)))!=len(hidden_wt): 
            print("duplicated weight_type!!!")
        
        register_hooks(self.encoder, hidden_wt, activation, False)
        register_hooks(self.decoder, hidden_wt, activation, False)        

		###### end 
        if segSize is None:
            if self.deep_sup_scale is not None: # use deep supervision technique
                (sup_pred, sup_pred_deepsup) = self.decoder(self.encoder(sup_feed_dict['img_data']\
				, return_feature_maps=True))
                if seq_feed_dict != None:
                    (seq_pred, seq_pred_deepsup) = self.decoder(self.encoder(seq_feed_dict['img_data'],
                        return_feature_maps=True))
            else:
                sup_pred = self.decoder(self.encoder(sup_feed_dict['img_data'], \
				return_feature_maps=True))
                if seq_feed_dict != None:
                    seq_pred = self.decoder(self.encoder(seq_feed_dict['img_data'],\
					return_feature_maps=True))


            loss = self.crit(sup_pred, sup_feed_dict['seg_label']) #this would be our sup loss

            MSELoss = nn.MSELoss()
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)

            def cal_weight(tensor, l):
                weights = [] 
                b, c, w, h = tensor.shape
                ind = 0 # the index of the image in the sequence with gt
                for i in range(l):
                    if i % seq_len == 0:
                        ind = i
                    #weights.append(torch.sum(cos(tensor[i], tensor[ind]))/(w * h))
                    weights.append(torch.sum(cos(torch.sum(tensor[i], dim=0), torch.sum(tensor[ind], dim=0)))/(w * h))
                return weights

            #weight_types = ['eta', 'cbr', 'conv_last']
            #weight_type = weight_types[2]

            if "share" in self.training_type:
                seq_losses = self.crit(seq_pred, seq_feed_dict['seg_label'])
                loss += seq_losses * epoch_weight 

            elif "seq" in self.training_type:
                ### all of this is for eta
                l = len(seq_feed_dict['seg_label'])
                seq_len = l / self.batch_size 

                # loss for each individual image
                losses = [self.crit(seq_pred[i,:,:,:].unsqueeze(0), seq_feed_dict['seg_label'][i,:,:].unsqueeze(0)) 
                        for i in range(l)]
                '''
                for i in range(l):
                    cv2.imwrite(str(i) + '.jpg', seq_feed_dict['img_data'][i].detach().cpu().numpy().transpose(1,2,0)*40)
                    cv2.imwrite(str(i) + '.png',seq_feed_dict['seg_label'][i].detach().cpu().numpy()*20)
                    cv2.imwrite(str(i) +'cbr.png', torch.sum(activation['cbr'][i], dim=0).detach().cpu().numpy())
                    cv2.imwrite(str(i) +'conv.png', torch.sum(activation['conv_last'][i], dim=0).detach().cpu().numpy()*-10)
                    cv2.imwrite(str(i) +'output.png', torch.sum(seq_pred[i], dim=0).detach().cpu().numpy()*-10)
                import bpython
                bpython.embed(locals())
                exit()
                '''
                mse_losses = []
                mse_losses2 = []
                cbr_losses = []
                conv_last_losses = []
		
                """
                used the (number of equal pixels in both seq_pred and gt label)/(total number pixel in the image) 
                as the weight = similarity level
                """
                # to change the weigh to one for images with actual gt labels
                l = len(losses)
                weights = []
                tensor = seq_pred # lets have the eta as the default (eta is when the weights are calculated based on the network's predictions)
                #if weight_type == 'eta':
                #    tensor = seq_pred
                
                '''
                # when the weights are only calculated based on the cbr layers in the decoder
                if weight_type == 'cbr':
                    tensor = activation['cbr']				

                # when the weights are only calculated based on the conv_last layers in the decoder
                if weight_type == 'conv_last':
                    tensor = activation['conv_last']				
                '''
                
                if len(activation.keys())==1: tensor = activation[list(activation.keys())[0]]
                
                # need to fix 
                # stack layer's weights
                if "-stack" in hidden_wt:
                    tmp = 1
                    for k, v in activation.items():
                        if tmp==1: 
                            tensor = v
                            if not isinstance(tensor, list): tensor = [tensor]
                            #print("tensor size: {}".format(len(tensor)))
                            tmp+=1
                        else:
                            if isinstance(v, list): 
                                tensor.extend(v)
                            else:
                                tensor.extend([v])
                            #print("tensor size: {}; appended {}".format(len(tensor), k))
                            
                            '''
                            if isinstance(v, list):
                                for i in range(len(v)): print("   {} shape: {}".format(i, v[i].shape))
                            else:
                                print("   shape: {}".format(v.shape))
                            '''
                # weights = cal_weight(tensor, l)
                eta_weights = cal_weight(seq_pred, l)
                
                '''
                for encoder hidden layers, hrnet's output is a list of tensors. calculate similarity weights for each of them, 
                then "mean stack"
                '''
                hidden_weights = []
                if isinstance(tensor, list):
                    tmp = []
                    for i in range(len(tensor)):
                        tmp.append(cal_weight(tensor[i], l))               
                    zipped_weights = zip(*tmp)
                    for w in zipped_weights:
                        hidden_weights.append(torch.mean(torch.stack(w))) 
                else:
                    hidden_weights = cal_weight(tensor, l)
                    
                    
                # when the weights are only calculated based on the predictions of the network and conv/cbr layers in the decoder
                '''
                -eta means combine seq_pred based weights and layer based weights 
                '''
                #if weight_type == 'eta-conv' or weight_type == 'eta-cbr': 
                if "-eta" in hidden_wt: 
                    #eta_weights = weights
                    #weights = []
                    '''
                    if weight_type == 'eta-conv':
                        tensor = activation['conv_last']				
                    if weight_type == 'eta-cbr':
                        tensor = activation['cbr']				
                    decoder_weights = cal_weight(tensor, l)
                    '''
                    zipped_weights = zip(eta_weights, hidden_weights)
                    for w in zipped_weights:
                        weights.append(torch.mean(torch.stack(w))) 
                else:
                    weights = hidden_weights

                #import bpython
                #bpython.embed(locals())
                #exit()
                
                weighted_losses = [a*b for a,b in zip(losses, weights)]
                #instead of averaging the loss for all sup and unsup togather, I separated them
                unsup_weighted_losses = []
                sup_weighted_losses = []
                for i in range(len(weighted_losses)):
                    if i % seq_len != 0:
                        unsup_weighted_losses.append(weighted_losses[i])
                    else:
                        sup_weighted_losses.append(weighted_losses[i])
                if len(unsup_weighted_losses) >= 1:
                    unsup_loss = torch.mean(torch.stack(unsup_weighted_losses))
                if len(sup_weighted_losses) >= 1:
                    sup_loss = torch.mean(torch.stack(sup_weighted_losses))
                ##elif len(unsup_weighted_losses) == 1:
                ##    unsup_loss = unsup_weighted_losses[0]
                
                '''
                ## for mse 
                if len(mse_losses) > 0:
                    mse_loss = torch.mean(torch.stack(mse_losses)) 
                loss += sup_loss +  mse_loss * epoch_weight
                '''
                unsup_loss = unsup_loss * epoch_weight
                loss += sup_loss + unsup_loss


            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(sup_pred_deepsup, sup_feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale

            acc = self.pixel_acc(sup_pred, sup_feed_dict['seg_label']) 
            return loss, acc #, weights, unsup_weighted_losses, unsup_loss
        # inference
        else:
            pred = self.decoder(self.encoder(sup_feed_dict['img_data'], return_feature_maps=True), segSize=segSize)
            return pred


class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    @staticmethod
    def build_encoder(arch='resnet50dilated', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)
        elif arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34dilated':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        elif arch == 'hrnetv2':
            net_encoder = hrnet.__dict__['hrnetv2'](pretrained=pretrained)
        elif arch == 'xception':
            net_encoder = xception.__dict__['xception'](pretrained=pretrained)
        elif arch == 'xception65':
            # This implementation of xception65 doesn't have imagenet weights 
            net_encoder = xception65.__dict__['xception65'](pretrained=pretrained)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='ppm_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        arch = arch.lower()
        if arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1':
            net_decoder = C1(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm':
            net_decoder = PPM(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]


# last conv, deep supervision
class C1DeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# last conv
class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x


# pyramid pooling
class PPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x
