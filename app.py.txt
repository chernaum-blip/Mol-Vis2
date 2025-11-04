import io
import gzip
from typing import List, Tuple, Set
import streamlit as st
import py3Dmol
from Bio.PDB import PDBList, PDBParser

st.set_page_config(page_title="Protein–Ligand H-Bond Explorer", layout="wide")

# ---------------- UI: sidebar controls ----------------
st.sidebar.title("Protein–Ligand H-Bond Explorer")
pdb_id_default = st.sidebar.text_input("PDB ID (e.g. 1HEW, 3PTB, 6Y2F)", "3PTB").strip().upper()
uploaded = st.sidebar.file_uploader("...or upload a .pdb file", type=["pdb"])
cutoff = st.sidebar.slider("H-bond cutoff (Å)", min_value=2.5, max_value=4.0, value=3.5, step=0.1)
use_tube = st.sidebar.checkbox("Use tube fallback (if cartoon fails)", value=False)
show_disulfides = st.sidebar.checkbox("Show disulfide bonds (SG–SG ≤ 2.2 Å)", value=True)
run_btn = st.sidebar.button("Render")

st.title("🧬 Protein–Ligand H-Bond Explorer")
st.caption("Ribbon protein + ligand sticks + automatic hydrogen bonds and residue labels.")

# ---------------- Data fetch ----------------
def fetch_pdb_text(pdb_id: str) -> str:
    pdbl = PDBList()
    path = pdbl.retrieve_pdb_file(pdb_id, pdir='.', file_format='pdb')
    if path.endswith(".gz"):
        with gzip.open(path, 'rt') as f:
            return f.read()
    with open(path) as f:
        return f.read()

def parse_structure_atoms(pdb_text: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("STRUCT", io.StringIO(pdb_text))
    protein_atoms, ligand_atoms, cys_sg = [], [], []
    for model in structure:
        for chain in model:
            for res in chain:
                resn = res.get_resname().strip().upper()
                het  = res.id[0].strip()
                if resn == "HOH":
                    continue
                if het == "":  # polymer / protein
                    for a in res:
                        protein_atoms.append(a)
                        if resn == "CYS" and a.get_id().upper() == "SG":
                            cys_sg.append(a)
                else:          # ligands & other het groups
                    for a in res:
                        ligand_atoms.append(a)
    return protein_atoms, ligand_atoms, cys_sg, structure

def elem_of(a):
    e = getattr(a, "element", "").strip()
    return e if e else a.get_id().strip()[:1]

def d2(a, b):
    v = a.get_vector() - b.get_vector()
    return v * v

def bonded_residue_key(atom):
    res = atom.get_parent()
    return (res.get_parent().id.strip(), int(res.id[1]), res.get_resname().strip().upper())

def compute_hbonds_and_ss(pdb_text: str, cutoff_ang: float = 3.5):
    protein_atoms, ligand_atoms, cys_sg, structure = parse_structure_atoms(pdb_text)
    lig_NO   = [a for a in ligand_atoms if elem_of(a) in ("N","O")]
    prot_NOS = [a for a in protein_atoms if elem_of(a) in ("N","O","S")]
    cut2 = cutoff_ang ** 2

    hbonds: List[Tuple] = []
    bonded_residues: Set[Tuple[str,int,str]] = set()
    for la in lig_NO:
        for pa in prot_NOS:
            if d2(la, pa) <= cut2:
                hbonds.append((la, pa))
                bonded_residues.add(bonded_residue_key(pa))

    ss_pairs = []
    ss2 = 2.2 ** 2
    for i in range(len(cys_sg)):
        for j in range(i+1, len(cys_sg)):
            if d2(cys_sg[i], cys_sg[j]) <= ss2:
                ss_pairs.append((cys_sg[i], cys_sg[j]))

    return hbonds, bonded_residues, ss_pairs

def polymer_chains_from_pdb_text(pdb_text: str):
    chains, seen = [], set()
    for ln in pdb_text.splitlines():
        if ln.startswith("ATOM"):
            ch = ln[21].strip() or "A"
            if ch not in seen:
                seen.add(ch); chains.append(ch)
    return chains

def render_view(pdb_text: str, cutoff_ang: float = 3.5, use_tube=False, show_ss=True):
    hbonds, bonded_residues, ss_pairs = compute_hbonds_and_ss(pdb_text, cutoff_ang=cutoff_ang)

    chains = polymer_chains_from_pdb_text(pdb_text)
    view = py3Dmol.view(width=1000, height=700)
    view.setBackgroundColor("#111731")
    view.addModel(pdb_text, "pdb")

    if use_tube:
        for ch in (chains or ['A']):
            view.setStyle({"chain": ch}, {"tube": {"radius": 0.5, "color": "spectrum"}})
    else:
        for ch in (chains or ['A']):
            view.setStyle({"chain": ch}, {"cartoon": {"color": "spectrum"}})

    # Ligands (non-water) as sticks
    view.setStyle({"hetflag": True, "not": {"resn": "HOH"}},
                  {"stick": {"colorscheme": "cyanCarbon", "radius": 0.28}})

    # H-bond lines
    for la, pa in hbonds:
        L = {"x": float(la.coord[0]), "y": float(la.coord[1]), "z": float(la.coord[2])}
        P = {"x": float(pa.coord[0]), "y": float(pa.coord[1]), "z": float(pa.coord[2])}
        view.addLine({"start": L, "end": P, "dashed": True, "color": "yellow", "linewidth": 2})

    # Disulfides
    if show_ss:
        for a, b in ss_pairs:
            A = {"x": float(a.coord[0]), "y": float(a.coord[1]), "z": float(a.coord[2])}
            B = {"x": float(b.coord[0]), "y": float(b.coord[1]), "z": float(b.coord[2])}
            view.addLine({"start": A, "end": B, "color": "green", "linewidth": 3})

    # Highlight & label bonded residues
    for ch, resi, resn in bonded_residues:
        view.setStyle({"chain": ch, "resi": int(resi)}, {"stick": {"radius": 0.3, "color": "magenta"}})
        view.addResLabels({"chain": ch, "resi": int(resi)},
                          {"fontColor": "white", "fontSize": 12, "backgroundOpacity": 0.6})

    view.zoomTo({"sel": "protein"})
    return view, hbonds, bonded_residues, ss_pairs

# ---------------- Load input ----------------
if uploaded is not None:
    pdb_text = uploaded.read().decode("utf-8", errors="ignore")
    source = f"Uploaded file: {uploaded.name}"
else:
    pdb_id = pdb_id_default if pdb_id_default else "3PTB"
    pdb_text = fetch_pdb_text(pdb_id)
    source = f"RCSB PDB ID: {pdb_id}"

# Trigger render automatically or on button
if (uploaded is not None) or run_btn or ("_ran_once" not in st.session_state):
    st.session_state["_ran_once"] = True

    with st.spinner(f"Rendering from {source} ..."):
        view, hbonds, bonded_residues, ss_pairs = render_view(
            pdb_text, cutoff_ang=cutoff, use_tube=use_tube, show_ss=show_disulfides
        )

    # Embed viewer
    html = view._make_html()
    st.components.v1.html(html, height=720, scrolling=False)

    # Right panel: data summary + CSV
    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader("Hydrogen-bonded residues (protein side)")
        if bonded_residues:
            rows = [{"chain": ch, "resi": resi, "resn": resn} for (ch, resi, resn) in sorted(bonded_residues, key=lambda x: (x[0], x[1]))]
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.info("No H-bonded residues found at this cutoff. Try a larger cutoff.")

    with col2:
        st.subheader("H-bond pairs (atom–atom)")
        if hbonds:
            rows = []
            for la, pa in hbonds:
                dist = ((la.get_vector() - pa.get_vector()) * (la.get_vector() - pa.get_vector())) ** 0.5
                res = pa.get_parent()
                chain = res.get_parent().id.strip()
                rows.append({
                    "ligand_atom": la.get_id(),
                    "protein_atom": pa.get_id(),
                    "protein_resn": res.get_resname().strip().upper(),
                    "protein_resi": int(res.id[1]),
                    "protein_chain": chain,
                    "distance_A": float(dist),
                })
            st.dataframe(rows, hide_index=True, use_container_width=True)

            # CSV download
            import pandas as pd
            df = pd.DataFrame(rows)
            st.download_button(
                "Download H-bond table (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="hbonds.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No H-bonds detected at current cutoff.")

    if show_disulfides:
        st.caption(f"Disulfide pairs detected: {len(ss_pairs)}")

st.markdown("""---
**Tips**  
• Type a PDB ID on the left or upload a `.pdb` file.  
• Adjust the hydrogen-bond cutoff to explore more/less interactions.  
• Use *Use tube fallback* if cartoons don’t render on your device.
""")
