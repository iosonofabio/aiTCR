import pathlib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1, seq3


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Emulate TCRen functionality")
    parser.add_argument(
        "--distance_threshold",
        default=5.0,
        type=float,
        help="Distance threshold in Angstroms",
    )
    parser.add_argument("--plot", action="store_true", help="Plot results")
    args = parser.parse_args()

    # Read the biophysics weight matrix (TCRen "potential")
    potential_fn = (
        pathlib.Path(__file__).parent.parent.parent
        / "software"
        / "tcren-ms"
        / "TCRen_potential.csv"
    )
    potential = pd.read_csv(potential_fn, index_col=[0, 1]).squeeze().unstack()

    import time

    t0 = time.time()

    # Define the path to the directory containing the TCRen files
    tcren_pdb_fn = (
        pathlib.Path(__file__).parent.parent.parent
        / "software"
        / "tcren-ms"
        / "example"
        / "input_structures"
        / "6uk4_TCRpMHCmodels_polyV.pdb"
    )

    with open(tcren_pdb_fn, "r") as f:
        pdb_parser = PDBParser(QUIET=True)
        structure = pdb_parser.get_structure("6uk4", f)

    # Split by chain
    chaind = {}
    for chain in structure.get_chains():
        # Get only TCR and peptide, the rest does not matter
        if chain.id in ("A", "B", "P"):
            chaind[chain.id] = chain

    # Split by residue
    atom_df = []
    for chname, chain in chaind.items():
        for ires, residue in enumerate(chain.get_residues()):
            resname = residue.get_resname()
            resnum = residue.id[1]
            for iatom, atom in enumerate(residue.get_atoms()):
                atom_df.append(
                    {
                        "chain": chname,
                        "residue": resname,
                        "pos": resnum,
                        "x": atom.coord[0],
                        "y": atom.coord[1],
                        "z": atom.coord[2],
                    }
                )
    atom_df = pd.DataFrame(atom_df)

    # Compute interchain dist for all atom pairs
    chain2 = "P"
    atom2 = atom_df[atom_df["chain"] == chain2]
    dis_df = []
    for chain1 in ("A", "B"):
        atom1 = atom_df[atom_df["chain"] == chain1]

        mg2, mg1 = np.meshgrid(np.arange(len(atom2)), np.arange(len(atom1)))

        dis_dfi = atom2.iloc[mg2.ravel()].copy()
        dis_dfi.rename(columns={"x": "x2", "y": "y2", "z": "z2"}, inplace=True)
        dis_dfi.rename(
            columns={"chain": "chain2", "residue": "residue2", "pos": "pos2"},
            inplace=True,
        )
        for col in ["chain", "residue", "pos", "x", "y", "z"]:
            dis_dfi[col] = atom1.iloc[mg1.ravel()][col].values

        dis_dfi["distance"] = np.sqrt(
            (dis_dfi["x"] - dis_dfi["x2"]) ** 2
            + (dis_dfi["y"] - dis_dfi["y2"]) ** 2
            + (dis_dfi["z"] - dis_dfi["z2"]) ** 2
        )
        dis_df.append(dis_dfi)
    dis_df = pd.concat(dis_df, ignore_index=True)

    # Compute min dist by residue and chain
    dismin = dis_df.groupby(
        ["chain", "pos", "residue", "chain2", "pos2", "residue2"]
    ).min()

    # Filter by threshold (5A)
    dismin_idx = dismin["distance"] < args.distance_threshold
    dismin_filt = dismin[dismin_idx].reset_index()

    # Contact matrix
    contact_matrix_bool = dismin_idx.unstack(["chain2", "pos2", "residue2"])

    # Weights
    contact_matrix_weighted = contact_matrix_bool.astype(float)
    for i, (chaini, posi, resi) in enumerate(contact_matrix_weighted.index.values):
        for j, (chainj, posj, resj) in enumerate(
            contact_matrix_weighted.columns.values
        ):
            if contact_matrix_weighted.iloc[i, j]:
                contact_matrix_weighted.iloc[i, j] *= potential.at[
                    seq1(resi), seq1(resj)
                ]

    # Array of contact weights
    contact_weights = contact_matrix_weighted.values.ravel()[
        contact_matrix_bool.values.ravel()
    ]

    # Final score
    score = contact_weights.sum()

    print(f"The TCRen score for this structure and epitope is: {score}.")
    t1 = time.time()
    print("The assessment took {:.2f} seconds.".format(t1 - t0))

    if args.plot:
        print("Plot contact matrices")
        cm_dict = {
            "bool": contact_matrix_bool,
            "weighted": contact_matrix_weighted,
        }
        for cmkind, contact_matrix in cm_dict.items():
            fig, axg = plt.subplots(
                2,
                2,
                figsize=(4, 20),
                gridspec_kw={
                    "height_ratios": [40, 1],
                    "width_ratios": [1, 20],
                },
            )
            axmain = axg[0, 1]

            kwargs = {}
            if cmkind == "weighted":
                vmax = np.abs(contact_matrix.values).max()
                kwargs["vmin"] = -vmax
                kwargs["vmax"] = vmax
                kwargs["cmap"] = "bwr"
                title = f"Total score: {contact_matrix.values.sum().round(2)}"
            else:
                kwargs = {"cmap": "Greys"}
                title = f"Contaxt matrix (<{args.distance_threshold} A)"
            axmain.matshow(contact_matrix, aspect="auto", **kwargs)

            aa_chain = contact_matrix.index.get_level_values("chain") == "A"
            axg[0, 0].matshow(np.array(aa_chain, ndmin=2).T, cmap="RdBu", aspect="auto")

            axg[1, 0].set_visible(False)
            axg[0, 0].sharey(axmain)
            axg[1, 1].sharex(axmain)
            axmain.tick_params(labelleft=False)
            axmain.tick_params(labelbottom=False)
            axmain.tick_params(labeltop=False)
            axg[0, 0].set_ylabel(
                "$\\beta$" + " " * 40 + "Chain/position" + " " * 40 + "$\\alpha$"
            )
            axg[0, 0].set_xticks([])
            axg[1, 1].set_xlabel("Peptide position")
            axg[1, 1].set_ylim(0, 1)
            for j, res2 in enumerate(
                contact_matrix.columns.get_level_values("residue2")
            ):
                axg[1, 1].text(
                    j,
                    0.5,
                    res2,
                    ha="center",
                    va="center",
                    rotation=0,
                    fontsize=8,
                )
            fig.tight_layout(h_pad=0, w_pad=0, rect=(0.03, 0.03, 0.97, 0.97))
            axmain.set_title(title)

        print("Plot potential")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        pmax = np.abs(potential.values).max()
        art = ax.matshow(
            potential.values, cmap="bwr", aspect=1.0, vmin=-pmax, vmax=pmax
        )
        cax = fig.colorbar(art, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("TCRen potential")
        ax.set_xticks(np.arange(len(potential.columns)))
        ax.set_yticks(np.arange(len(potential.index)))
        ax.set_xticklabels(potential.columns)
        ax.set_yticklabels(potential.index)
        ax.set_xlabel("Peptide residue")
        ax.set_ylabel("$\\alpha$ / $\\beta$ TCR residue")
        fig.tight_layout()

        print("Plot cumulative of contact weights")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.ecdf(contact_weights, complementary=True)
        ax.set_xlabel("Contact score")
        ax.set_ylabel("Cumulative fraction")
        ax.axvline(0, lw=2, color="k")
        ax.grid(True)
        fig.tight_layout()

        plt.ion()
        plt.show()
