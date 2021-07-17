modes = ["gen", "dis"]

# training settings
batch_size_gen = 64  # batch size for the generator
batch_size_dis = 64  # batch size for the discriminator
lr_gen = 1e-6  # learning rate for the generator
lr_dis = 1e-6  # learning rate for the discriminator

lambda_gen = 1e-5  # l2 loss regulation weight for the generator
lambda_dis = 1e-5  # l2 loss regulation weight for the discriminator

n_uu_sample_gen = 20  # number of samples for the generator
n_ua_sample_gen = 20
n_au_sample_gen = 20

n_epochs = 25  # number of outer loops
n_epochs_gen = 5  # number of inner loops for the generator
n_epochs_dis = 10  # number of inner loops for the discriminator
display = 1
if_teacher_forcing = True
n_epochs_gen_teacher = 5

gen_interval = n_epochs_gen  # sample new nodes for the generator for every gen_interval iterations
dis_interval = n_epochs_dis  # sample new nodes for the discriminator for every dis_interval iterations
update_ratio = 1    # updating ratio when choose the trees

# other hyper-parameters
n_emb = 100  # dim
multi_processing = False  # whether using multi-processing to construct BFS-trees
window_size = 2

app = "basic_evaluation"
dataset = "Flickr"
pretrain_mode = "-PV-DBOW"

label_filename = "../../data/" + dataset + "_node_label"

# model saving
load_model = False  # whether loading existing model for initialization
save_steps = 10
model_log = "../../save_model/" + dataset + "/"

# path settings
uu_filename = "../../data/" + dataset + "_uu.txt"
ua_filename = "../../data/" + dataset + "_ua.txt"
au_filename = "../../data/" + dataset + "_au.txt"

train_filename = "../../data/" + app + "/" + dataset + "_train.txt"
test_filename = "../../data/" + app + "/" + dataset + "_test.txt"
test_neg_filename = "../../data/" + app + "/" + dataset + "_test_neg.txt"

pretrain_node_emb_filename_d = "../../pre_train/" + dataset + pretrain_mode + "_node_pre_train.mat"
pretrain_att_emb_filename_d = "../../pre_train/" + dataset + pretrain_mode + "_att_pre_train.mat"
pretrain_node_emb_filename_g = "../../pre_train/" + dataset + pretrain_mode + "_node_pre_train.mat"
pretrain_att_emb_filename_g = "../../pre_train/" + dataset + pretrain_mode + "_att_pre_train.mat"
fir_sim_label_filename = "../../data/" + dataset + "_fir_sim_label.mat"
sed_sim_label_filename = "../../data/" + dataset + "_sed_sim_label.mat"
att_sim_label_filename = "../../data/" + dataset + "_att_sim_label.mat"

is_trained_pretrain = False
trained_node_emb_filename_d = "../../pre_train/" + dataset + pretrain_mode + "_node_pre_train.mat"
trained_att_emb_filename_d = "../../pre_train/" + dataset + pretrain_mode + "_att_pre_train.mat"
trained_node_emb_filename_g = "../../pre_train/" + dataset + pretrain_mode + "_node_pre_train.mat"
trained_att_emb_filename_g = "../../pre_train/" + dataset + pretrain_mode + "_att_pre_train.mat"


emb_filenames = ["../../results/" + app + "-" + dataset + "_gen_",
                 "../../results/" + app + "-" + dataset + "_dis_"]
result_filename = "../../results/" + app + "-" + dataset + pretrain_mode + ".txt"
cache_filename = "../../cache/" + dataset

