#################################################
##### Superconductivity Featurizer Notebook #####
#################################################
# This python file imports superconductivity compositions (stored in ../data/supercon.csv) and uses matminer to extract features  
# and data from the composition. This data is then exported to ../data/supercon_features.csv, after which machine learning models 
# can be trained to predict the Tc of superconductors based on composition.
#
# Author: Kirk Kleinsasser
#################################################

#################################################
################## Run Imports ##################
#################################################
# %%
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matminer.datasets import get_available_datasets, load_dataset, get_all_dataset_info
from matminer.featurizers.conversions import StrToComposition

parser = argparse.ArgumentParser(description="A program that gets feature from composition of superconductor critical temperatures.")
parser.add_argument('-f', '--file', action='store', default="supercon_dataset.csv", dest='filename', help='Specify filename to featurize. ')
args = parser.parse_args()
filename = args.filename

#load supercon databse and metadata
#print(get_all_dataset_info("superconductivity2018")) #print metadata
#features will be made with matminer, target is Tc
data = pd.DataFrame(pd.read_csv(f'./data/{filename}'))
stc = StrToComposition()
composition = stc.featurize_dataframe(data, 'name', ignore_errors=True)
composition.head()

#################################################
################ Get Featurizers ################
#################################################
# %% 
from matminer.featurizers.composition import ElectronAffinity, ElementProperty, TMetalFraction, Stoichiometry, BandCenter, OxidationStates, IonProperty, ElectronegativityDiff, AtomicOrbitals, ValenceOrbital, AtomicPackingEfficiency, CohesiveEnergy, CohesiveEnergyMP
#WARNING - this will take a while to run!
#works but returns nan


#these are all the Magpie features. You can choose fewer if you want.
features = ['Number', 'MendeleevNumber', 'AtomicWeight', 'MeltingT', 
            'Column', 'Row', 'CovalentRadius', 'Electronegativity', 
            'NsValence', 'NpValence', 'NdValence', 'NfValence', 'NValence', 
            'NsUnfilled', 'NpUnfilled', 'NdUnfilled', 'NfUnfilled', 'NUnfilled', 
            'GSvolume_pa', 'GSbandgap', 'GSmagmom', 'SpaceGroupNumber']

from matminer.featurizers.composition import ElectronAffinity, ElementProperty, TMetalFraction, Stoichiometry, BandCenter, OxidationStates, IonProperty, ElectronegativityDiff, AtomicOrbitals, ValenceOrbital, AtomicPackingEfficiency, CohesiveEnergy, CohesiveEnergyMP
#WARNING - this will take a while (>2 hours) to run!

#these are all the Magpie features. You can choose fewer if you want.
features = ['Number', 'MendeleevNumber', 'AtomicWeight', 'MeltingT', 
            'Column', 'Row', 'CovalentRadius', 'Electronegativity', 
            'NsValence', 'NpValence', 'NdValence', 'NfValence', 'NValence', 
            'NsUnfilled', 'NpUnfilled', 'NdUnfilled', 'NfUnfilled', 'NUnfilled', 
            'GSvolume_pa', 'GSbandgap', 'GSmagmom', 'SpaceGroupNumber']

ea_prop = ElementProperty(data_source='magpie', features=features, stats=['mean']) # can add 'minimum', 'maximum', 'range', 'avg_dev', 'mode' to stats
ea_prop_data = ea_prop.featurize_dataframe(composition, 'composition', ignore_errors=True)

el_prop = ElementProperty(data_source='magpie', features=features, stats=['mean'])
el_prop_data = el_prop.featurize_dataframe(composition, 'composition', ignore_errors=True)

ea_aff = ElectronAffinity()
ea_aff_data = ea_aff.featurize_dataframe(composition, 'composition', ignore_errors=True)

met_frac = TMetalFraction()
met_frac_data = met_frac.featurize_dataframe(composition, 'composition', ignore_errors=True)

stoich = Stoichiometry()
stoich_data = stoich.featurize_dataframe(composition, 'composition', ignore_errors=True)

band_center = BandCenter()
band_center_data = band_center.featurize_dataframe(composition, 'composition', ignore_errors=True)

ox_states = OxidationStates()
ox_states_data = ox_states.featurize_dataframe(composition, 'composition', ignore_errors=True)

ion_prop = IonProperty()
ion_prop_data = ion_prop.featurize_dataframe(composition, 'composition', ignore_errors=True)

en_diff = ElectronegativityDiff()
en_diff_data = en_diff.featurize_dataframe(composition, 'composition', ignore_errors=True)

atom_orbitals = AtomicOrbitals()
atom_orbitals_data = atom_orbitals.featurize_dataframe(composition, 'composition', ignore_errors=True)

val_orbitals = ValenceOrbital()
val_orbitals_data = val_orbitals.featurize_dataframe(composition, 'composition', ignore_errors=True)

atom_pack_eff = AtomicPackingEfficiency()
atom_pack_eff_data = atom_pack_eff.featurize_dataframe(composition, 'composition', ignore_errors=True)

cohesive_en = CohesiveEnergy()
cohesive_en_data = cohesive_en.featurize_dataframe(composition, 'composition', ignore_errors=True)

cohesive_en_mp = CohesiveEnergyMP()
cohesive_en_mp_data = cohesive_en_mp.featurize_dataframe(composition, 'composition', ignore_errors=True)

#################################################
################## Export Data ##################
#################################################
# %%
all_feat = ea_aff_data
for featurizer in [el_prop_data, met_frac_data, stoich_data, band_center_data, ox_states_data, ion_prop_data, en_diff_data, atom_orbitals_data, val_orbitals_data, atom_pack_eff_data, cohesive_en_data, cohesive_en_mp_data]:
    all_feat = pd.merge(all_feat, featurizer, how="left") #merges each featurizer with the main set, but drops any duplicate columns (otherwise we'd have many Tc and name columns)
all_feat.to_csv('../data/supercon_features.csv') #export features for use in juypter

# dill.dump_session('../data/latest-run.db') #dump python session for external use

#################################################