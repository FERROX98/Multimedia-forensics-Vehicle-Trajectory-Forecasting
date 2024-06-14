from __future__ import division
import torch
import torch.nn as nn


"""
This is Model class for the HighwayNet model, which is a LSTM based model
for vehicle trajectory forecasting.
and inspired by :
https://github.com/nachiket92/conv-social-pooling/blob/master
"""


class highwayNet(nn.Module):

    # Initialization
    def __init__(self, args):
        super(highwayNet, self).__init__()

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
            args["in_target_embedding_size"], args["target_encoder_size"], 1, dropout=0.2
        )

        # Encoder LSTM
        self.nbrs_encoder_lstm = torch.nn.LSTM(
            args["in_nbrs_embedding_size"], args["nbrs_encoder_size"], 1,  dropout=0.2
        )

        # Decoder LSTM
        self.dec_lstm = torch.nn.LSTM(
            args["nbrs_encoder_size"] + args["target_encoder_size"],
            args["decoder_size"], dropout=0.2
        )

        # Output layers:
        self.output_layer = torch.nn.Linear(args["decoder_size"], args["output_size"])
        self.relu = torch.nn.ReLU()

    def forward(self, hist, nbrs):
        # [ seq/time -> (15 (3s * 5Fps) + current time t = 16 ), b_size, feat_size (2->x,y)]
        if self.debug:
            print("hist.shape: ", hist.shape)
            print("nbrs.shape: ", nbrs.shape)
            
        fc_out = self.leaky_relu(self.input_target_layer(hist))
        _, (hist_enc, _) = self.target_enc_lstm(fc_out)

        # TODO Griglia vicini mockata
        nbrs_enc = torch.zeros_like(hist_enc).float()
        if self.debug:
            print("nbrs_enc.shape: ", nbrs_enc.shape)
            print("hist_enc.shape: ", hist_enc.shape)

        encoder = torch.cat((hist_enc, nbrs_enc), 2)
        if self.debug:
            print("enc.shape: ", encoder.shape)

        fut_pred = self.decode(encoder)

        return fut_pred

    def decode(self, encoder):

        encoder = encoder.repeat(self.args["n_predicted_frame"], 1, 1)
        if (self.debug): 
            print("enc.shape: ", encoder.shape)
            
        h_dec, _ = self.dec_lstm(encoder)
        if (self.debug): 
            print("h_dec.shape: ", h_dec.shape)
            
        # h_dec = h_dec.permute(1, 0, 2)
        # if (self.debug): 
        #     print("h_dec.shape2: ", h_dec.shape)
            
        fut_pred = self.output_layer(h_dec)
        if (self.debug): 
            print("fut_pred.shape: ", fut_pred.shape)
            
        # fut_pred = self.relu(fut_pred)
        # if (self.debug): 
        #     print("fut_pred.shape3: ", fut_pred.shape)
            
        return fut_pred
    
    def gaussian_bivariate_distribution(self,out):
        
        mux = out[:,:,1]
        muy = out[:,:,2]
        varx = out[:,:,3]
        vary = out[:,:,4]
        rho = torch.tanh(out[:,:,5]) # because covariance must be between -1 and 1 in order to not have negative determinant (det(cov) = varx*vary - (rho)^2) 
        return torch.cat((mux,muy,varx,vary,rho),dim=2)
        
        
    
    
    # def decode_by_step(self, enc):

    #     pre_traj = []

    #     decoder_input = enc

    #     for _ in range(self.out_length):
    #         decoder_input = decoder_input.unsqueeze(0)
    #         h_dec, _ = self.dec_lstm(decoder_input)
    #         h_for_pred = h_dec.squeeze()
    #         fut_pred = self.op(h_for_pred)
    #         pre_traj.append(fut_pred.view(fut_pred.size()[0], -1))

    #         embedding_input = fut_pred
    #         decoder_input = self.spatial_embedding(embedding_input)

    #     pre_traj = torch.stack(pre_traj, dim=0)
    #     pre_traj = outputActivation(pre_traj)
    #     return pre_traj
