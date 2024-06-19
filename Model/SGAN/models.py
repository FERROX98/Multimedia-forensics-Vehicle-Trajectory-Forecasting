from __future__ import division
import sys
import torch
import torch.nn as nn
import torch
import torch.nn as nn

"""
This is Model class for the HighwayNet model, which is a LSTM based model
for vehicle trajectory forecasting.
and inspired by :
https://github.com/nachiket92/conv-social-pooling/blob/master
and 
https://github.com/agrimgupta92/sgan
"""


class highwayNetDiscriminator(nn.Module):
    def __init__(
        self,
        obs_len=15,
        pred_len=25,
        embedding_dim=256,
        bottleneck_dim=1024,
        h_dim=128,
    ):
        super(highwayNetDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.n_levels = 1
        self.encoder = nn.LSTM(embedding_dim, h_dim, self.n_levels)

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.relu = torch.nn.ReLU()
        self.real_classifier = nn.Sequential(
             nn.Linear(h_dim, bottleneck_dim, 1), nn.LeakyReLU(0.1), nn.Dropout(p=0.2),
             nn.Linear(bottleneck_dim, 1, 1), nn.Sigmoid(),
        )

    def init_hidden(self, batch):
        return (
            torch.zeros(self.n_levels, batch, self.h_dim).cuda(),
            torch.zeros(self.n_levels, batch, self.h_dim).cuda(),
        )

    def forward(self, traj):
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
        _, (h_state, _) = self.encoder(obs_traj_embedding, state_tuple)
        classifier_input = h_state.squeeze()

        scores = self.real_classifier(classifier_input)
        return scores


class highwayNetGenerator(nn.Module):

    # Initialization
    def __init__(
        self,
        debug = False,
        train_flag = False,
        in_target_length=2,
        in_target_embedding_size=128,
        in_grid_nbrs_length=3,
        in_nbrs_embedding_size=128,
        target_encoder_size=64,
        dyn_embedding_size = 1024, 
        nbrs_encoder_size=64,
        decoder_size=128,
        output_size=2,
    ):
        super(highwayNetGenerator, self).__init__()

        # Use gpu flag
        self.use_cuda = True

        self.debug = debug

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = train_flag
        self.in_target_length = in_target_length
        self.in_target_embedding_size = in_target_embedding_size
        self.in_grid_nbrs_length = in_grid_nbrs_length
        self.in_nbrs_embedding_size = in_nbrs_embedding_size
        self.target_encoder_size = target_encoder_size
        self.nbrs_encoder_size = nbrs_encoder_size
        self.decoder_size = decoder_size
        self.output_size = output_size
        self.dyn_embedding_size = dyn_embedding_size
        self.layers_encoder = 2
        self.layers_nbrs_encoder = 2
        # Input embedding layer
        self.input_target_layer = nn.Linear(in_target_length, in_target_embedding_size)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

        self.input_nbrs_layer = nn.Linear(in_grid_nbrs_length, in_nbrs_embedding_size)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

        # Encoder LSTM
        self.target_enc_lstm = torch.nn.LSTM(
            in_target_embedding_size, target_encoder_size, self.layers_encoder, dropout=0.2
        )
        
        self.dyn_emb = torch.nn.Linear(self.target_encoder_size, self.dyn_embedding_size)

        # Encoder LSTM
        self.nbrs_encoder_lstm = torch.nn.LSTM(
            in_nbrs_embedding_size, nbrs_encoder_size, self.layers_nbrs_encoder, dropout=0.2
        )

        # Decoder LSTM
        self.dec_lstm = torch.nn.LSTM(
            self.dyn_embedding_size + target_encoder_size, decoder_size, 1
        )
        self.embending = torch.nn.Linear(self.decoder_size, self.dyn_embedding_size + target_encoder_size,1 )

        # Output layers:
        self.output_layer = torch.nn.Linear(decoder_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, hist, nbrs):
        # [ seq/time -> (15 (3s * 5Fps) + current time t = 16 ), b_size, feat_size (2->x,y)]
        if self.debug:
            print("hist.shape: ", hist.shape)
            print("nbrs.shape: ", nbrs.shape)
        if torch.isnan(hist).any() or torch.isnan(nbrs).any():
            print("NAN")

        batch = hist.size(1)

        fc_out_history = self.leaky_relu(self.input_target_layer(hist))
        state_tuple = self.init_hidden_enc(batch)
        _, (hist_enc, _) = self.target_enc_lstm(fc_out_history, state_tuple)
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc))

        # if torch.isnan(fc_out_history).any():
        #     print("NAN")
        # if torch.isnan(hist_enc).any():
        #     print("NAN")

      #  fc_out_nbrs = self.leaky_relu(self.input_target_layer(nbrs.view(-1, 2)))
        fc_out_nbrs = self.leaky_relu(self.input_target_layer(nbrs))

        # if torch.isnan(fc_out_nbrs).any():
        #     print("NAN")
        fc_out_nbrs = fc_out_nbrs.view(-1, batch, self.in_nbrs_embedding_size)
        # if torch.isnan(fc_out_nbrs).any():
        #     print("NAN")
        state_tuple = self.init_hidden_nbrs_enc(batch)
        _, (nbrs_enc, _) = self.nbrs_encoder_lstm(fc_out_nbrs, state_tuple)
        # if torch.isnan(nbrs_enc).any():
        #     print("NAN")
        encoder_h = torch.cat((hist_enc, nbrs_enc), 2)
        # if torch.isnan(encoder_h).any():
        #     print("NAN")
        # if self.debug:
        #     print("enc.shape: ", encoder_h.shape)

        fut_pred = self.decode(encoder_h)

        return fut_pred

    # Generator forward pass
    def decode(self, encoder_h):

        batch = encoder_h.size(1)
        pred_traj_fake_rel = []
        state_tuple = self.init_hidden(batch)

        for _ in range(25):

            output, state_tuple = self.dec_lstm(encoder_h, state_tuple)
            fut_pred = self.output_layer(state_tuple[0])
            encoder_h = self.embending(state_tuple[0])
            encoder_h = encoder_h.view(1, batch, -1)
            #state_tuple = (fut_pred, state_tuple[1])
            #fut_pred = self.gaussian_bivariate_distribution(fut_pred)


            pred_traj_fake_rel.append(fut_pred)

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=-1)
        pred_traj_fake_rel = pred_traj_fake_rel.squeeze(dim=0).permute(2, 0, 1)
        if torch.isnan(pred_traj_fake_rel).any():
            print("NAN")
        return pred_traj_fake_rel

    def init_hidden(self, batch):
        return (
            torch.zeros(1, batch, self.decoder_size).cuda(),
            torch.zeros(1, batch, self.decoder_size).cuda(),
        )
        
    def init_hidden_enc(self, batch):
        return (
            torch.zeros(self.layers_encoder, batch, self.target_encoder_size).cuda(),
            torch.zeros(self.layers_encoder, batch, self.target_encoder_size).cuda(),
        )
        
    def init_hidden_nbrs_enc(self, batch):
        return (
            torch.zeros(self.layers_nbrs_encoder, batch, self.nbrs_encoder_size).cuda(),
            torch.zeros(self.layers_nbrs_encoder, batch, self.nbrs_encoder_size).cuda(),
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
