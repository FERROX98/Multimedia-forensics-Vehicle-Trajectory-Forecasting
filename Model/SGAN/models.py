
from __future__ import division
import sys 
import torch
import torch.nn as nn
import torch
import torch.nn as nn




class highwayNetDiscriminator(nn.Module):
    def __init__(
        self,
        args,
        obs_len = 15,
        pred_len = 25,
        embedding_dim=64,
        h_dim=64,
        mlp_dim=1024,
    ):
        super(highwayNetDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.encoder = nn.LSTM(embedding_dim, h_dim, 1)

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.relu = torch.nn.ReLU()
        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(h_dim, 1, 1), nn.Sigmoid() )
        
    def init_hidden(self, batch):
        return (
            torch.zeros(1, batch, self.h_dim).cuda(),
            torch.zeros(1, batch, self.h_dim).cuda(),
        )

    def forward(self, traj, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        batch = traj.size(1)
        obs_traj_embedding = self.spatial_embedding(traj.view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        state_tuple = self.init_hidden(batch)
        _, (h_state,_) = self.encoder(obs_traj_embedding, state_tuple)
      
        
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.

        classifier_input = h_state.squeeze()
       
        scores = self.real_classifier(classifier_input)
        return scores




"""
This is Model class for the HighwayNet model, which is a LSTM based model
for vehicle trajectory forecasting.
and inspired by :
https://github.com/nachiket92/conv-social-pooling/blob/master
and 
https://github.com/agrimgupta92/sgan
"""
class highwayNetGenerator(nn.Module):

    # Initialization
    def __init__(self, args):
        super(highwayNetGenerator, self).__init__()

        self.args = args

        # Use gpu flag
        self.use_cuda = True

        self.debug = args["debug"]

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args["train_flag"]

        # Input embedding layer
        self.input_target_layer = nn.Linear(
            args["in_target_length"], args["in_target_embedding_size"]
        )
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

        self.input_nbrs_layer = nn.Linear(
            args["in_grid_nbrs_length"], args["in_nbrs_embedding_size"]
        )
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

        # Encoder LSTM
        self.target_enc_lstm = torch.nn.LSTM(
            args["in_target_embedding_size"],
            args["target_encoder_size"],
        )

        # Encoder LSTM
        self.nbrs_encoder_lstm = torch.nn.LSTM(
            args["in_nbrs_embedding_size"], args["nbrs_encoder_size"], 1
        )

        #Decoder LSTM
        self.dec_lstm = torch.nn.LSTM(
            args["nbrs_encoder_size"] + args["target_encoder_size"],
            args["decoder_size"]
        )

        # Output layers:
        self.output_layer = torch.nn.Linear(args["decoder_size"], args["output_size"])
        self.relu = torch.nn.ReLU()
     

    def forward(self, hist, nbrs):
        # [ seq/time -> (15 (3s * 5Fps) + current time t = 16 ), b_size, feat_size (2->x,y)]
        if self.debug:
            print("hist.shape: ", hist.shape)
            print("nbrs.shape: ", nbrs.shape)
        batch = hist.size(1)
        fc_out_history = self.leaky_relu(self.input_target_layer(hist))
        
        _, (hist_enc, _) = self.target_enc_lstm(fc_out_history)
    
 
        fc_out_nbrs = self.leaky_relu(self.input_target_layer(nbrs.view(-1, 2)))
        fc_out_nbrs = fc_out_nbrs.view(-1, batch,  self.args["in_nbrs_embedding_size"])
        _, (nbrs_enc, _) = self.target_enc_lstm(fc_out_nbrs)
       
        encoder_h = torch.cat((hist_enc, nbrs_enc), 2)
        if self.debug:
            print("enc.shape: ", encoder_h.shape)

        fut_pred = self.decode(encoder_h)

        return fut_pred

    # Generator forward pass
    def decode(self, encoder_h):

        batch = encoder_h.size(1)
        pred_traj_fake_rel = []
        state_tuple = self.init_hidden(batch)
        for _ in range(25):
            
            output, state_tuple = self.dec_lstm(encoder_h,state_tuple)
            fut_pred = self.output_layer(state_tuple[0])
            fut_pred = self.gaussian_bivariate_distribution(fut_pred)
                                                       #fut_pred[:, :, 1:2+1]
            #fut_pred = torch.squeeze(fut_pred, 0)
            pred_traj_fake_rel.append(fut_pred)

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=-1)
        pred_traj_fake_rel = pred_traj_fake_rel.permute(2, 1, 0)
        
        return pred_traj_fake_rel
    
    def init_hidden(self, batch):
        return (
            torch.zeros(1, batch, self.args["decoder_size"]).cuda(),
            torch.zeros(1, batch, self.args["decoder_size"]).cuda(),
        )
        
    def gaussian_bivariate_distribution(self, out):

        mux = out[:, :, 0]
        muy = out[:, :, 1]
        varx = out[:, :, 2]
        vary = out[:, :, 3]
        rho = torch.tanh(
            out[:, :, 4]
        )  # because covariance must be between -1 and 1 in ordert to not have negative determinant (det(cov) = varx*vary - (rho)^2) then
        return torch.cat((mux, muy, varx, vary, rho), dim=0)

 
if __name__ == "__main__":
    model_generator = highwayNetGenerator(load_args())
    
    model_discriminator = highwayNetDiscriminator(load_args())
