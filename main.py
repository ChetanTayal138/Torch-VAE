from model import Weights, Encoder, Decoder
from utils import xavier, latent_space
from loss import reconstruction_loss, kl_divergence_loss, network_loss
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

def forward_propogate(INPUT_IMAGE, weights, biases):

        mean, std = Encoder(INPUT_IMAGE).encode(weights, biases)
        latent_layer = latent_space(mean, std)
        decoder_output = Decoder(latent_layer).decode(weights, biases)

        return decoder_output, mean, std

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

    mnist_trainset = datasets.MNIST(root='./', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size = 32, shuffle = True)
    return trainloader
        

if __name__ == "__main__":

    trainloader = load_mnist()
    
    learning_rate = 0.00001
    epochs = 30000
    batch_size = 32

    IMAGE_DIM = 28 * 28
    NN_DIM = 512
    LATENT_SPACE_DIM = 2

    ALPHA = 0
    BETA = 0 


    # input is 1 x 784

    weight_list = {
            "w1": Weights("weight_matrix_encoder_hidden", [IMAGE_DIM,NN_DIM]), # 784 x 512 
            "w2": Weights("weight_mean_hidden", [NN_DIM, LATENT_SPACE_DIM]), # 512 x 2
            "w3": Weights("weight_std_hidden", [NN_DIM, LATENT_SPACE_DIM]), # 512 x 2
            "w4": Weights("weight_matrix_decoder_hidden",[LATENT_SPACE_DIM, NN_DIM]), # 2 x 512
            "w5": Weights("weight_decoder",[NN_DIM, IMAGE_DIM]) # 512 x 784
            }

    bias_list = {
            "b1": Weights("bias_matrix_encoder_hidden", [1,NN_DIM]), # 512
            "b2": Weights("bias_mean_hidden", [1,LATENT_SPACE_DIM]), # 2
            "b3": Weights("bias_std_hidden", [1,LATENT_SPACE_DIM]), # 2
            "b4": Weights("bias_matrix_decoder_hidden",[1,NN_DIM]), # 512
            "b5": Weights("bias_decoder",[1,IMAGE_DIM]) # 784
        }
    

    INPUT_IMAGE = torch.rand(32,IMAGE_DIM)

    for i in range(1,epochs):

        for batch, _ in tqdm(trainloader):
            batch = batch.view(batch.shape[0], -1)
            
            decoder_output, mean, std = forward_propogate(batch, weight_list, bias_list)
            total_loss = reconstruction_loss(batch, decoder_output) +  kl_divergence_loss(mean, std)
            total_loss.sum().backward()
            
            #print(weight_list["w1"]._get_weight().grad.data)
            for weight in weight_list:
                #print(weight_list[weight])
                weight_list[weight]._get_weight().data = weight_list[weight]._get_weight().data - learning_rate * weight_list[weight]._get_weight().grad.data

            for bias in bias_list:
                bias_list[bias]._get_weight().data = bias_list[bias]._get_weight().data - learning_rate * bias_list[bias]._get_weight().grad.data
        
           

            for weight in weight_list:
                weight_list[weight]._get_weight().grad.data.zero_()

            for bias in bias_list:
                bias_list[bias]._get_weight().grad.data.zero_()

        if i % 2 == 0:
            print("Total Loss : ", total_loss.sum().data)
        
