import numpy as np
from rdkit import Chem

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                       'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                       'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',    # H?
                                       'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                       'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()])

def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))


def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))


def gen_adj_matrix(mol):
    """
         Generate adjacency matrix of mol
         Return : numpy matrix [#atom * #atom]
    """
    n_atoms = mol.GetNumAtoms()
    adj_mat = np.zeros([n_atoms, n_atoms])
    bonds = mol.GetBonds()
    for bond in bonds:
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        adj_mat[start, end] = 1
        adj_mat[end, start] = 1
    adj_mat = adj_mat + np.eye(n_atoms)    
    return adj_mat.reshape([n_atoms * n_atoms])


def genAtomFeatureMatrix(mol):
    """ 
        Generate feature matrix of atoms in mol
        Return : numpy matrix [ #atom * Ndim_atom_feature]
    """
    n_atoms = mol.GetNumAtoms()
    Ndim_atom = num_atom_features()
    atom_f = np.zeros([n_atoms, Ndim_atom])

    for i, atom in enumerate(mol.GetAtoms()):
        atom_f[i, :] = atom_features(atom)

    return atom_f.reshape([n_atoms * Ndim_atom])


def getnBondFeatureMatrix(mol):
    """
        Generate Bond feature matrix 
        Return : numpy ndarray [#atoms * #atoms * Ndim_bond_feature]
    """
    n_atoms = mol.GetNumAtoms()
    Ndim_bond = num_bond_features() 
    bond_f = np.zeros([n_atoms, n_atoms, Ndim_bond])
    for i, bond in enumerate(mol.GetBonds()):
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        bond_f[atom1, atom2, :] = bond_features(bond)
        bond_f[atom2, atom1, :] = bond_features(bond)
    return bond_f.reshape([n_atoms * n_atoms * Ndim_bond])


def gen_feature_vector(mol):
    """
        Generate feature vector of mol
        feature_vector : [ #atom , feature matrix([#atom * Ndim_atom_feature]), 
                            adjacency matrix([#atom, #atom]), 
                            bond feature([#atom, #atom, Ndim_bond_feature])]  
    """
    n_atoms = np.array(mol.GetNumAtoms())
    atom_feature = genAtomFeatureMatrix(mol)
    adj_mat = gen_adj_matrix(mol)
    bond_feature = getnBondFeatureMatrix(mol)
    return np.hstack([n_atoms, atom_feature, adj_mat, bond_feature])


if __name__ == '__main__':
    mol = Chem.MolFromSmiles('CC(N)C')
    vec = gen_feature_vector(mol)
    print(vec)
    print(vec.shape)


    sm = [x.strip() for x in open("./smiles.txt").readlines()]
    data_raw = [ gen_feature_vector(Chem.MolFromSmiles(x)) for x in sm]
    max_len = max([x.shape for x in data_raw])[0]
    print(max_len)
    data = np.zeros([len(data_raw), max_len])
    for i, row in enumerate(data_raw):
        data[i,:row.shape[0]] = row
    
    data = np.vstack(data)
    print(data.shape)
    np.save("data",data)

