data = load('ValSet.mat');
variables = fieldnames(data);
traj = data.res_traj
traj = traj(1:2,1:2)
track = data.res_traj
track = track(1:2,1:2)
fname = 'light.mat'; 
save(fname,'traj','track')