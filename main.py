from model import Weights, Encoder, Decoder
from utils import xavier, latent_space
from loss import reconstruction_loss, kl_divergence_loss, network_loss
import torch


def forward_propogate(INPUT_IMAGE, weights, biases):

        mean, std = Encoder(INPUT_IMAGE).encode(weights, biases)
        latent_layer = latent_space(mean, std)
        decoder_output = Decoder(latent_layer).decode(weights, biases)

        return decoder_output, mean, std

        

if __name__ == "__main__":

    learning_rate = 0.001
    epochs = 3
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
            "b1": Weights("bias_matrix_encoder_hidden", [NN_DIM]), # 512
            "b2": Weights("bias_mean_hidden", [LATENT_SPACE_DIM]), # 2
            "b3": Weights("bias_std_hidden", [LATENT_SPACE_DIM]), # 2
            "b4": Weights("bias_matrix_decoder_hidden",[NN_DIM]), # 512
            "b5": Weights("bias_decoder",[IMAGE_DIM]) # 784
        }
    

    INPUT_IMAGE = torch.randn(32,IMAGE_DIM)

    for i in range(epochs):

        decoder_output, mean, std = forward_propogate(INPUT_IMAGE, weight_list, bias_list)
        total_loss = reconstruction_loss(INPUT_IMAGE, decoder_output) +  kl_divergence_loss(mean, std)
        total_loss.sum().backward()

        print(weight_list["w1"]._get_weight().grad.data)

        for weight in weight_list:
            print(weight_list[weight])
            weight_list[weight]._get_weight().grad.data = weight_list[weight]._get_weight().grad.data - learning_rate * weight_list[weight]._get_weight().grad.data

        #for bias in bias_list:
        #    bias_list[bias] = bias_list[bias] - learning_rate * bias_list[bias]._get_weight().grad.data

    
        print(weight_list["w1"]._get_weight().grad.data)
