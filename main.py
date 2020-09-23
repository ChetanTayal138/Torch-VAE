from model import Weights, Encoder, Decoder
from utils import xavier, latent_layer
import torch


if __name__ == "__main__":

    learning_rate = 0.001
    epochs = 3000
    batch_size = 32

    IMAGE_DIM = 28 * 28
    NN_DIM = 512
    LATENT_SPACE_DIM = 2


    
    weight_list = {
            "w1": Weights("weight_matrix_encoder_hidden", [IMAGE_DIM,NN_DIM]),
            "w2": Weights("weight_mean_hidden", [NN_DIM, LATENT_SPACE_DIM]),
            "w3": Weights("weight_std_hidden", [NN_DIM, LATENT_SPACE_DIM]),
            "w4": Weights("weight_matrix_decoder_hidden",[LATENT_SPACE_DIM, NN_DIM]),
            "w5": Weights("weight_decoder",[NN_DIM, IMAGE_DIM])
            }

    bias_list = {
            "b1": Weights("bias_matrix_encoder_hidden", [IMAGE_DIM,NN_DIM]),
            "b2": Weights("bias_mean_hidden", [NN_DIM, LATENT_SPACE_DIM]),
            "b3": Weights("bias_std_hidden", [NN_DIM, LATENT_SPACE_DIM]),
            "b4": Weights("bias_matrix_decoder_hidden",[LATENT_SPACE_DIM, NN_DIM]),
            "b5": Weights("bias_decoder",[NN_DIM, IMAGE_DIM])
        }
    

    INPUT_IMAGE = torch.randn(NN_DIM)
    mean, std = Encoder(INPUT_IMAGE).encode(weight_list, bias_list)

    print(mean.size())
    print(std.size())

    
    
