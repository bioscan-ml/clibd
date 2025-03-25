from Bio import Entrez
import time
import scipy.io as sio
import json
from tqdm import tqdm

# Please enter your email here

Entrez.email = None

def get_species_to_other_level_dict(species_list, taxonomic_info=None):
    failed_species = []

    if taxonomic_info is None:
        taxonomic_info = {}
    pbad = tqdm(species_list)
    for species in pbad:
        if species in taxonomic_info.keys():
            continue
        try:
            handle = Entrez.esearch(db="taxonomy", term=species)
            record = Entrez.read(handle)
            handle.close()

            if record["IdList"]:
                tax_id = record["IdList"][0]

                handle = Entrez.efetch(db="taxonomy", id=tax_id, retmode="xml")
                records = Entrez.read(handle)
                handle.close()

                for record in records:
                    info = {}
                    for lineage in record['LineageEx']:
                        if lineage['Rank'] in ['genus', 'family', 'order']:
                            info[lineage['Rank']] = lineage['ScientificName']

                    for level in ['genus', 'family', 'order']:
                        if level not in info.keys():
                            info[level] = 'not_classified'

                    taxonomic_info[species] = info
                    pbad.set_description(
                        f"order: {taxonomic_info[species]['order']} || family: {taxonomic_info[species]['family']} || genus: {taxonomic_info[species]['genus']}  || species: {species}")
            else:
                taxonomic_info[species] = {}
                taxonomic_info[species]['order'] = 'not_classified'
                taxonomic_info[species]['family'] = 'not_classified'
                taxonomic_info[species]['genus'] = 'not_classified'
        except:
            taxonomic_info[species] = {}
            taxonomic_info[species]['order'] = 'not_classified'
            taxonomic_info[species]['family'] = 'not_classified'
            taxonomic_info[species]['genus'] = 'not_classified'


        time.sleep(0.5)

    return taxonomic_info, failed_species

def convert_species_nd_array_to_list(ndarray):
    list_of_strings = [arr[0][0] for arr in ndarray]
    print(list_of_strings[0])

    return list_of_strings

if __name__ == '__main__':

    if Entrez.email is None:
        raise ValueError('Please enter your email in the Entrez.email variable')

    filename = 'specie_to_other_labels.json'
    with open(filename, 'r') as file:
        specie_to_other_labels = json.load(file)

    # load all species of INSECT dataset.
    att_splits_mat = sio.loadmat('att_splits.mat')
    res101_mat = sio.loadmat('res101.mat')

    all_species = list(set(convert_species_nd_array_to_list(res101_mat['species'])))

    specie_to_other_labels, failed_species = get_species_to_other_level_dict(all_species, taxonomic_info=specie_to_other_labels)



    with open(filename, 'w') as f:
        json.dump(specie_to_other_labels, f, indent=4)

    print("Done")

    print(f'Dictionary has been saved to {filename}')
    if len(failed_species) > 0:
        print(f'failed with following species:')
        print(failed_species)

