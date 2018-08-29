import vgg
import inception_v3

net_dict = {
    "vgg": vgg,
    "inception_v3": inception_v3
}

def get_basenet(name, inputs):
    net = net_dict[name];
    return net.basenet(inputs);
