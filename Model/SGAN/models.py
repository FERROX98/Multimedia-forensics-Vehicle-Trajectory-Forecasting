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
            nn.Linear(h_dim, bottleneck_dim, 1),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2),
            nn.Linear(bottleneck_dim, 1, 1),
            nn.Sigmoid(),
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
        debug=False,
        
        in_size_hist=4,
        out_size_hist=64,
        
        in_size_vel=1,
        in_size_acc=1,
        out_emb_vel_acc=32,
        
        in_grid_nbrs_size=2,
        out_grid_nbrs_size=4,
        
        out_size_encoder_hist=64,
        out_size_emb_target=128,
        
        out_middle_emb_nbrs_enc=4,
        out_nbrs_encoder_size=128,
        
        out_size_pre_dec=64,
        out_size_dec=256,
        
        output_size=2,
    ):
        super(highwayNetGenerator, self).__init__()

        self.debug = debug

        # input vel acc
        self.in_size_vel = in_size_vel
        self.in_size_acc = in_size_acc
        self.in_size_vel_acc = in_size_vel + in_size_acc
        self.out_emb_vel_acc = out_emb_vel_acc

        # in history
        self.in_size_hist = in_size_hist
        self.out_size_hist = out_size_hist

        # combine acc + vel + history encoder
        self.layers_target_encoder = 2
        self.in_size_encoder_hist = out_size_hist
        self.out_size_encoder_hist = out_size_encoder_hist

        # middle mlp target enc
        self.in_size_emb_target = out_size_encoder_hist
        self.out_size_emb_target = out_size_emb_target

        # pre enc nbrs
        self.in_grid_nbrs_size = in_grid_nbrs_size
        self.out_grid_nbrs_size = out_grid_nbrs_size

        # encode nbrs
        self.layers_nbrs_encoder = 2
        self.in_nbrs_encoder_size = out_grid_nbrs_size
        self.out_nbrs_encoder_size = out_nbrs_encoder_size

        # middle mlp nbrs enc
        self.in_middle_emb_nbrs_enc = out_nbrs_encoder_size
        self.out_middle_emb_nbrs_enc = out_middle_emb_nbrs_enc

        # pre dec
        self.in_size_pre_dec = out_size_emb_target + out_emb_vel_acc + out_middle_emb_nbrs_enc
        self.out_size_pre_dec = out_size_pre_dec

        # decoder
        self.in_size_dec = out_size_pre_dec
        self.out_size_dec = out_size_dec

        # final output x,y
        self.in_last_mlp = out_size_dec
        self.out_last_mlp = output_size

        # pre enc history
        self.input_target_layer = nn.Sequential(
            nn.Linear(self.in_size_hist, self.out_size_hist), 
       #     torch.nn.LeakyReLU(0.1)
        )

        # enc history
        self.target_enc_lstm = torch.nn.LSTM(
            self.in_size_encoder_hist,
            self.out_size_encoder_hist,
            self.layers_target_encoder,
            dropout=0 if self.layers_target_encoder <2 else 0.2 ,
            bidirectional=False,
        )

        # middle mlp target enc
        self.middle_mlp_target_enc = nn.Sequential(
            torch.nn.Linear(self.in_size_emb_target, self.out_size_emb_target),
           # torch.nn.LeakyReLU(0.1),
        )

        # embedding vel acc
        self.input_target_vel_acc_layer = nn.Sequential(
            nn.Linear(self.in_size_vel_acc, self.out_emb_vel_acc),
            torch.nn.LeakyReLU(0.1),
        )

        # pre enc nbrs
        self.input_nbrs_layer = nn.Sequential(
            nn.Linear(self.in_grid_nbrs_size, self.out_grid_nbrs_size),
            torch.nn.LeakyReLU(0.1),
        )

        # enc nbrs
        self.nbrs_encoder_lstm = torch.nn.LSTM(
            self.in_nbrs_encoder_size,
            self.out_nbrs_encoder_size,
            self.layers_nbrs_encoder,
            dropout= 0 if self.layers_nbrs_encoder<2 else  0.2,
            bidirectional=False,
        )

        # middle mlp nbrs enc
        self.middle_mlp_nbrs_enc = nn.Sequential(
            torch.nn.Linear(self.in_middle_emb_nbrs_enc, self.out_middle_emb_nbrs_enc),
          #  torch.nn.LeakyReLU(0.1),
        )
        # pre dec
        self.pre_dec = nn.Sequential(
            nn.Linear(self.in_size_pre_dec, self.out_size_pre_dec),
            nn.LeakyReLU(0.1),
            #nn.Dropout(p=0.2),
        )

        # dec
        self.dec_lstm = torch.nn.LSTM(self.in_size_dec, self.out_size_dec, 1)

        # recurseve embedding decoder
        self.rec_embending_dec = torch.nn.Linear(self.out_size_dec, self.in_size_dec, 1)

        # Output layers:
        self.output_layer = nn.Sequential(
            nn.Linear(self.out_size_dec, self.out_size_dec),
          # nn.Linear(self.out_size_dec, self.in_last_mlp),
            #nn.ReLU(),
          #  nn.Linear(self.out_size_dec, self.out_size_dec),
            #torch.nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2),
            nn.Linear(self.out_size_dec, self.out_last_mlp),
        )

    # [ seq/time -> (15 (3s * 5Fps) + current time t = 16 ), b_size, feat_size (2->x,y)]
    def forward(self, hist, nbrs, vel, acc):

        vel = torch.unsqueeze(vel, dim=0)
        acc = torch.unsqueeze(acc, dim=0)

        batch = hist.size(1)

        # pre enc history
        pre_target_enc = self.input_target_layer(hist)

        # enc history
        state_tuple = self.init_hidden_enc(batch)
        _, (hist_enc, _) = self.target_enc_lstm(
            pre_target_enc, (state_tuple[0].detach(), state_tuple[1].detach())
        )

        # middle mlp target enc
        hist_enc = self.middle_mlp_target_enc(hist_enc)

        # concat vel acc
        vel_acc = torch.cat((vel, acc), 2)

        # embedding vel acc
        fc_out_vel_acc = self.input_target_vel_acc_layer(vel_acc)

        # concat history and vel acc
        hist_enc = torch.cat(
            (torch.unsqueeze(hist_enc[-1, :, :], dim=0), fc_out_vel_acc), 2
        )

        # pre enc nbrs
        fc_out_nbrs = self.input_nbrs_layer(nbrs)
        fc_out_nbrs = fc_out_nbrs.view(-1, batch, self.in_nbrs_encoder_size)

        # enc nbrs
        state_tuple = self.init_hidden_nbrs_enc(batch)
        _, (nbrs_enc, _) = self.nbrs_encoder_lstm(fc_out_nbrs, state_tuple)

        # middle mlp nbrs enc
        nbrs_enc = torch.unsqueeze(nbrs_enc[-1, :, :], dim=0)
        nbrs_enc = self.middle_mlp_nbrs_enc(nbrs_enc)

        # concat history and nbrs
        encoder_h = torch.cat((hist_enc, nbrs_enc), 2)

        # pre dec
        fut_pred = self.pre_dec(encoder_h)

        # dec
        fut_pred = self.decode(fut_pred)

        return fut_pred

    # Generator forward pass
    def decode(self, encoder_h):

        batch = encoder_h.size(1)
        pred_traj_fake_rel = []
        state_tuple = self.init_hidden(batch)

        for _ in range(25):
            state_tuple = self.init_hidden(batch)
            output, state_tuple = self.dec_lstm(encoder_h, state_tuple)

            # output
            fut_pred = self.output_layer(output)
           # fut_pred = self.gaussian_bivariate_distribution(fut_pred)
            
            # recursive embedding
            encoder_h = self.rec_embending_dec(state_tuple[0])
            encoder_h = encoder_h.view(1, batch, -1)

            pred_traj_fake_rel.append(fut_pred)

        if len(pred_traj_fake_rel[0]) == 2:
            for el in pred_traj_fake_rel:
                el = el.unsqueeze(dim=0)

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        pred_traj_fake_rel = pred_traj_fake_rel.squeeze(dim=0).view(-1, batch, 2)

        return pred_traj_fake_rel

    def init_hidden(self, batch):
        return (
            torch.zeros(1, batch, self.out_size_dec).cuda(),
            torch.zeros(1, batch, self.out_size_dec).cuda(),
        )

    def init_hidden_enc(self, batch):
        return (
            torch.zeros(self.layers_target_encoder * 1, batch, self.out_size_encoder_hist)
            .cuda()
            .requires_grad_(),
            torch.zeros(self.layers_target_encoder * 1, batch, self.out_size_encoder_hist)
            .cuda()
            .requires_grad_(),
        )

    def init_hidden_nbrs_enc(self, batch):
        return (
            torch.zeros(self.layers_nbrs_encoder * 1, batch, self.out_nbrs_encoder_size)
            .cuda()
            .requires_grad_(),
            torch.zeros(self.layers_nbrs_encoder * 1, batch, self.out_nbrs_encoder_size)
            .cuda()
            .requires_grad_(),
        )
        

    #(Graves, 2015)
    def gaussian_bivariate_distribution(self, out):

        mux = out[:, :, 0]
        muy = out[:, :, 1]
        varx = out[:, :, 2]
        vary = out[:, :, 3]
        varx = torch.exp(varx)
        vary = torch.exp(vary)
        rho = torch.tanh(
            out[:, :, 4]
        )  # because covariance must be between -1 and 1 in ordert to not have negative determinant (det(cov) = varx*vary - (rho)^2) then
        return torch.cat((mux, muy, varx, vary, rho), dim=2)



if __name__ == "__main__":
    model_generator = highwayNetGenerator(load_args())

    model_discriminator = highwayNetDiscriminator(load_args())
