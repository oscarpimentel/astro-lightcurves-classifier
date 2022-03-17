from __future__ import print_function
from __future__ import division
from __future__ import annotations
from . import _C

from fuzzytools.datascience.grid_search import GDIter, GridSeacher
from . import model_baselines as mbls

MAX_DAY = _C.MAX_DAY
ENC_NOF_LAYERS = 1

# RNN_CELLS = ['GRU']
# RNN_CELLS = ['LSTM']
RNN_CELLS = ['GRU', 'LSTM']
USES_OBSE_INPUT = False
#TE_FEATURES = [0*2, 4*2, 8*2]
TE_FEATURES = [0*2]
# HEADS = [0, 8, 16]
# HEADS = [2]
HEADS = [0]

DEFAUL_TE_FEATURES = 12*2
DEFAUL_KERNEL_SIZE = 1
DEFAULT_TIME_NOISE = '6*24**-1'
DEFAUL_HEADS = 4

###################################################################################################################################################

class ModelCollections():
	def __init__(self, lcdataset):
		self.lcdataset = lcdataset
		self.max_day = MAX_DAY

		bands = 2
		d = 64 # 16 32 64 128
		self.gd_embd_dims = GDIter(d*bands)
		self.enc_nof_layers = ENC_NOF_LAYERS

		p = 0/100
		self.dropout_d = {
			'p':p,
			'r':p,
			}
		self.common_dict = {
			'max_period':self.max_day*1.5, # ***
			'band_names':lcdataset[lcdataset.get_lcset_names()[0]].band_names,
			'output_dims':len(lcdataset[lcdataset.get_lcset_names()[0]].class_names),
			}
		self.base_dict = {
			'class_mdl_kwargs':{
				'layers':2, # 1 2
				'dropout':{
					'p':50/100,
					},
				},
			}
		self.reset()

	def reset(self):
		self.mps = []

	def __repr__(self):
		txt = ''
		for k,mp in enumerate(self.mps):
			txt += f'({k}) - mdl_kwargs: {mp["mdl_kwargs"]}\n'
			txt += f'({k}) - dataset_kwargs: {mp["dataset_kwargs"]}\n'
			txt += f'({k}) - class_mdl_kwargs: {mp["class_mdl_kwargs"]}\n'
			txt += f'---\n'
		return txt

	def add_gs(self, gs):
		new_mps = []
		gs.update({
			'dataset_kwargs':{
				'in_attrs':['obs']+(['obse'] if USES_OBSE_INPUT else []),
				'rec_attr':'obs',
				'max_day':self.max_day,
			}})
		mps = gs.get_dicts()
		for mp in mps:
			bd = self.base_dict.copy()
			bd.update(mp)
			new_mps += GridSeacher(bd).get_dicts()

		for mp in new_mps:
			for k in self.common_dict.keys():
				for k2 in mp.keys():
					if k2!='dataset_kwargs':
						mp[k2][k] = self.common_dict[k]
		self.mps += new_mps

###################################################################################################################################################

	def s_attn_model(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialTimeModSelfAttnModel,
				'embd_dims':self.gd_embd_dims,
				'layers':self.enc_nof_layers,
				'dropout':self.dropout_d,

				'te_features':GDIter(DEFAUL_TE_FEATURES),
				'kernel_size':GDIter(DEFAUL_KERNEL_SIZE),
				'time_noise_window':GDIter(DEFAULT_TIME_NOISE),
				'heads':GDIter(DEFAUL_HEADS),
			},
		})
		self.add_gs(gs)

	def s_attn_models_te(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialTimeModSelfAttnModel,
				'embd_dims':self.gd_embd_dims,
				'layers':self.enc_nof_layers,
				'dropout':self.dropout_d,

				'te_features':GDIter(*TE_FEATURES), # ***
				'kernel_size':GDIter(DEFAUL_KERNEL_SIZE),
				'time_noise_window':GDIter(DEFAULT_TIME_NOISE),
				'heads':GDIter(DEFAUL_HEADS),
			},
		})
		self.add_gs(gs)

	def s_attn_models_heads(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialTimeModSelfAttnModel,
				'embd_dims':self.gd_embd_dims,
				'layers':self.enc_nof_layers,
				'dropout':self.dropout_d,

				'te_features':GDIter(DEFAUL_TE_FEATURES),
				'kernel_size':GDIter(DEFAUL_KERNEL_SIZE),
				'time_noise_window':GDIter(DEFAULT_TIME_NOISE),
				'heads':GDIter(*HEADS), # ***
			},
		})
		self.add_gs(gs)

	def s_attn_models_dummy(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialTimeModSelfAttnModel,
				'embd_dims':self.gd_embd_dims,
				'layers':self.enc_nof_layers,
				'dropout':self.dropout_d,

				'te_features':GDIter(0),
				'kernel_size':GDIter(DEFAUL_KERNEL_SIZE),
				'time_noise_window':GDIter(DEFAULT_TIME_NOISE),
				'heads':GDIter(0), # ***
			},
		})
		self.add_gs(gs)

	def s_attn_extra_models(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialTimeCatSelfAttnModel,
				'embd_dims':self.gd_embd_dims,
				'layers':self.enc_nof_layers,
				'dropout':self.dropout_d,

				'te_features':GDIter(DEFAUL_TE_FEATURES),
				'kernel_size':GDIter(DEFAUL_KERNEL_SIZE),
				'time_noise_window':GDIter(DEFAULT_TIME_NOISE),
				'heads':GDIter(DEFAUL_HEADS), # ***
			},
		})
		self.add_gs(gs)

###################################################################################################################################################

	def p_attn_model(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelTimeModSelfAttnModel,
				'embd_dims':self.gd_embd_dims,
				'layers':self.enc_nof_layers,
				'dropout':self.dropout_d,

				'te_features':GDIter(DEFAUL_TE_FEATURES),
				'kernel_size':GDIter(DEFAUL_KERNEL_SIZE),
				'time_noise_window':GDIter(DEFAULT_TIME_NOISE),
				'heads':GDIter(DEFAUL_HEADS),
			},
		})
		self.add_gs(gs)

	def p_attn_models_te(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelTimeModSelfAttnModel,
				'embd_dims':self.gd_embd_dims,
				'layers':self.enc_nof_layers,
				'dropout':self.dropout_d,

				'te_features':GDIter(*TE_FEATURES), # ***
				'kernel_size':GDIter(DEFAUL_KERNEL_SIZE),
				'time_noise_window':GDIter(DEFAULT_TIME_NOISE),
				'heads':GDIter(DEFAUL_HEADS),
			},
		})
		self.add_gs(gs)

	def p_attn_models_heads(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelTimeModSelfAttnModel,
				'embd_dims':self.gd_embd_dims,
				'layers':self.enc_nof_layers,
				'dropout':self.dropout_d,

				'te_features':GDIter(DEFAUL_TE_FEATURES),
				'kernel_size':GDIter(DEFAUL_KERNEL_SIZE),
				'time_noise_window':GDIter(DEFAULT_TIME_NOISE),
				'heads':GDIter(*HEADS), # ***
			},
		})
		self.add_gs(gs)

	def p_attn_models_dummy(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelTimeModSelfAttnModel,
				'embd_dims':self.gd_embd_dims,
				'layers':self.enc_nof_layers,
				'dropout':self.dropout_d,

				'te_features':GDIter(0),
				'kernel_size':GDIter(DEFAUL_KERNEL_SIZE),
				'time_noise_window':GDIter(DEFAULT_TIME_NOISE),
				'heads':GDIter(0), # ***
			},
		})
		self.add_gs(gs)

	def p_attn_extra_models(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelTimeCatSelfAttnModel,
				'embd_dims':self.gd_embd_dims,
				'layers':self.enc_nof_layers,
				'dropout':self.dropout_d,

				'te_features':GDIter(DEFAUL_TE_FEATURES),
				'kernel_size':GDIter(DEFAUL_KERNEL_SIZE),
				'time_noise_window':GDIter(DEFAULT_TIME_NOISE),
				'heads':GDIter(DEFAUL_HEADS), # ***
			},
		})
		self.add_gs(gs)


###################################################################################################################################################

	def s_rnn_models(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialRNNModel,
				'rnn_cell_name':GDIter(*RNN_CELLS),
				'embd_dims':self.gd_embd_dims,
				'layers':self.enc_nof_layers,
				'dropout':self.dropout_d,
			},
		})
		self.add_gs(gs)

	def s_rnn_extra_models(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.SerialTimeModRNNModel,
				'rnn_cell_name':GDIter(*RNN_CELLS),
				'embd_dims':self.gd_embd_dims,
				'layers':self.enc_nof_layers,
				'dropout':self.dropout_d,

				'te_features':GDIter(DEFAUL_TE_FEATURES),
				'kernel_size':GDIter(DEFAUL_KERNEL_SIZE),
				'time_noise_window':GDIter(DEFAULT_TIME_NOISE),
			},
		})
		self.add_gs(gs)

###################################################################################################################################################

	def p_rnn_models(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelRNNModel,
				'rnn_cell_name':GDIter(*RNN_CELLS),
				'embd_dims':self.gd_embd_dims,
				'layers':self.enc_nof_layers,
				'dropout':self.dropout_d,
			},
		})
		self.add_gs(gs)

	def p_rnn_extra_models(self):
		gs = GridSeacher({
			'mdl_kwargs':{
				'C':mbls.ParallelTimeModRNNModel,
				'rnn_cell_name':GDIter(*RNN_CELLS),
				'embd_dims':self.gd_embd_dims,
				'layers':self.enc_nof_layers,
				'dropout':self.dropout_d,

				'te_features':GDIter(DEFAUL_TE_FEATURES),
				'kernel_size':GDIter(DEFAUL_KERNEL_SIZE),
				'time_noise_window':GDIter(DEFAULT_TIME_NOISE),
			},
		})
		self.add_gs(gs)

