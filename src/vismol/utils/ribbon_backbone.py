#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  ribbon_backbone.py
#
#  Independent Cartoon representation -- DATA layer (chain-aware backbone
#  extraction + secondary-structure classification). See this project's
#  own plan file for the full design rationale (user's own request: a
#  from-scratch cartoon representation, not influenced by the existing
#  vismol.utils.cartoon_BCK.py it replaces).
#
#  [EN] Deliberately does NOT reuse VismolObject.define_Calpha_backbone()
#  (vismol_object.py) -- that method flattens EVERY chain's CA atoms into
#  ONE global list and only detects a backbone break via consecutive PDB
#  residue numbering (residue.index == previous.index + 1) plus a < 4.0 A
#  distance sanity check, NEVER comparing chain identity -- two different
#  chains whose PDB numbering happens to run consecutively (common) can
#  get bridged by a bogus CA-CA "bond". extract_backbone_chains() below
#  fixes that class of bug by construction: chain identity is the PRIMARY
#  segmentation key, residue-numbering continuity is only ever checked
#  WITHIN an already-chain-scoped run.
#
#  classify_secondary_structure() implements the same published P-SEA
#  (Labesse et al. 1997) Ca-only H/S/C classification thresholds the
#  (being-replaced) cartoon_BCK.py's own calculate_secondary_structure()
#  used -- the THRESHOLDS are cited science, not borrowed code; this is a
#  fresh, independent implementation, operating on one already-chain-
#  scoped Ca array at a time (never a cross-chain-merged list), with its
#  own dihedral/angle helpers and a simple minimum-run-length smoothing
#  pass instead of whatever block-merging the old module did internally.
# ============================================================================
import numpy as np


def extract_backbone_chains ( vismol_object, frame = None ):
    """ Returns a list of per-chain backbone records, one per CONTIGUOUS
    protein run:
        { "chain": Chain, "residues": [Residue, ...],
          "ca": np.ndarray (N,3), "o": [np.ndarray(3) or None, ...] }

    A residue starts a NEW record whenever (a) it belongs to a different
    Chain object than the one currently running, or (b) its own PDB
    residue.index isn't exactly `previous_residue.index + 1` (a real gap --
    missing residues, a ligand/HETATM block, ...) -- so a chain with an
    internal gap correctly produces TWO separate ribbon segments, not one
    falsely-continuous one bridging the gap.

    Residues that aren't recognised as protein (residue.is_protein False --
    ligands, waters, ions; see Residue._is_protein(), reused as-is here, a
    boolean CLASSIFIER not a rendering concern) or that are missing a CA
    atom simply close out whatever record is running and are skipped, same
    as a chain change -- a ribbon cannot pass through a point that doesn't
    exist. Records shorter than 2 residues (nothing to draw a spline
    through) are dropped entirely.

    `frame` is passed straight to Atom.coords(); None (the default) uses
    whatever frame the session currently has active -- matching how every
    other "just show me the CURRENT structure" representation build
    already behaves in this codebase, NOT a live trajectory subscription
    (this feature's own plan documents that live per-frame rebuild is out
    of scope for v1). """
    records = [ ]
    current = None

    for chain in vismol_object.chains.values ( ):
        # residues is a dict keyed by PDB residue index -- iterate in that
        # NUMERIC order, not dict/insertion order (not guaranteed already
        # sorted for every loader).
        for residue_index in sorted ( chain.residues.keys ( ) ):
            residue = chain.residues[residue_index]

            ca_atom = None
            o_atom  = None
            if residue.is_protein:
                for atom in residue.atoms.values ( ):
                    if atom.name == "CA":
                        ca_atom = atom
                    elif atom.name == "O":
                        o_atom = atom

            if ca_atom is None:
                current = None
                continue

            starts_new = (
                current is None
                or current["chain"] is not chain
                or residue.index != current["residues"][-1].index + 1
            )
            if starts_new:
                current = { "chain": chain, "residues": [ ], "_ca": [ ], "o": [ ] }
                records.append ( current )

            current["residues"].append ( residue )
            current["_ca"].append ( ca_atom.coords ( frame ) )
            current["o"].append ( o_atom.coords ( frame ) if o_atom is not None else None )

    result = [ ]
    for record in records:
        if len ( record["residues"] ) < 2:
            continue
        record["ca"] = np.asarray ( record["_ca"], dtype = np.float64 )
        del record["_ca"]
        result.append ( record )
    return result


def _dihedral ( p0, p1, p2, p3 ):
    """ Signed dihedral angle (radians, [-pi, pi]) of the p0-p1-p2-p3
    quadruplet -- standard atan2-of-cross/dot formulation (numerically
    stable at angles near 0/180 degrees, unlike an arccos-of-dot
    formulation would be). """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm ( b1 )
    if b1_norm < 1e-9:
        return 0.0
    b1 = b1 / b1_norm
    v = b0 - np.dot ( b0, b1 ) * b1
    w = b2 - np.dot ( b2, b1 ) * b1
    x = np.dot ( v, w )
    y = np.dot ( np.cross ( b1, v ), w )
    return float ( np.arctan2 ( y, x ) )


def _angle_between ( v0, v1 ):
    """ Unsigned angle (radians, [0, pi]) between two vectors -- returns
    None if either is (near) zero-length (degenerate, e.g. two
    coincident Ca positions). """
    n0 = np.linalg.norm ( v0 )
    n1 = np.linalg.norm ( v1 )
    if n0 < 1e-9 or n1 < 1e-9:
        return None
    cos_t = np.clip ( np.dot ( v0, v1 ) / ( n0 * n1 ), -1.0, 1.0 )
    return float ( np.arccos ( cos_t ) )


def _signed_angle_delta_deg ( angle_deg, target_deg ):
    """ Shortest signed difference (degrees) between angle_deg and
    target_deg, correctly wrapping across the +-180 degree boundary (a
    plain subtraction would report ~360 degrees "off" for e.g. 179 vs.
    -179, when the true difference is 2 degrees). """
    return ( angle_deg - target_deg + 180.0 ) % 360.0 - 180.0


def _smooth_short_runs ( codes, min_run = 3 ):
    """ Demotes any H/S run shorter than `min_run` residues back to 'C' --
    a single-residue "H" or "S" surrounded by coil is almost always noise
    at these Ca-only thresholds, not a real secondary-structure element;
    a real helix/strand is at least a few residues long. Simple run-
    length pass, independent of whatever block-acceptance logic the
    module being replaced used internally. """
    codes = codes.copy ( )
    n = len ( codes )
    i = 0
    while i < n:
        j = i
        while j < n and codes[j] == codes[i]:
            j += 1
        if codes[i] != "C" and ( j - i ) < min_run:
            codes[i:j] = "C"
        i = j
    return codes


def classify_secondary_structure ( ca_coords, min_run = 2, helix_tolerance_scale = 1.0, strand_tolerance_scale = 1.5 ):
    """ P-SEA-style (Labesse et al. 1997) Ca-only H(elix)/S(trand)/C(oil)
    classification. `ca_coords` must already be a SINGLE chain's own
    contiguous Ca array (see extract_backbone_chains() above) -- this
    function itself has no chain-awareness of its own, by design, so it
    can never accidentally classify across a chain boundary.

    For each residue i (needs 2 residues of context on either side --
    i-1 and i+1..i+3 -- so only i in [1, n-4] are ever classified; the
    first/last few residues of any chain always come back 'C'), computes:
      - d2/d3/d4: Ca(i-1)-Ca(i+1)/(i+2)/(i+3) distances
      - t : bond angle at Ca(i), between the (i-1)->i and (i+1)->i
            directions (i.e. the Ca(i-1)-Ca(i)-Ca(i+1) angle)
      - a : dihedral of the Ca(i-1),Ca(i),Ca(i+1),Ca(i+2) quadruplet
    against the published threshold ranges:

                           Helix (published)     Strand (published)
      Angle t (deg)        89 +/- 12             124 +/- 14
      Dihedral a (deg)     50 +/- 20            -170 +/- 45
      Distance d2 (A)      5.5 +/- 0.5            6.7 +/- 0.6
      Distance d3 (A)      5.3 +/- 0.5            9.9 +/- 0.9
      Distance d4 (A)      6.4 +/- 0.6           12.4 +/- 1.1

    Every "+/-" tolerance above is multiplied by `helix_tolerance_scale` /
    `strand_tolerance_scale` respectively (uniformly across all 5 criteria
    for that SS type) BEFORE comparing -- 1.0 = the published values
    as-is, >1.0 = more permissive (more residues classified as that SS
    type), <1.0 = stricter.

    [EN] TOLERANCE WIDENED (user report: "a tolerancia para considerar ou
    nao uma estrutura em fita esta muito rigorosa"): strand_tolerance_
    scale's own DEFAULT is 1.5 (i.e. the published values +50%), not 1.0,
    since real beta strands are frequently a little distorted/twisted
    relative to the idealized geometry the paper's own thresholds were
    fit to, and the published tolerances were visibly rejecting real
    strand residues as coil. helix_tolerance_scale defaults to 1.0 (only
    strand detection was reported as too strict). Both are now also
    user-adjustable live from Preferences > Cartoon (see DEFAULT_CARTOON_
    SETTINGS in ribbon_geometry.py and setup_interface.py's
    __apply_cartoon_parameters()) -- the user's own follow-up request for
    "um criterio variavel para a atribuicao de helices e fitas". min_run's
    own default (see signature) was also lowered from 3 to 2 for the same
    original reason.

    then runs _smooth_short_runs() to drop isolated single-residue H or S
    calls back to coil. Returns a numpy array of single-character '<U1'
    codes, same length as ca_coords. """
    n = len ( ca_coords )
    codes = np.full ( n, "C", dtype = "<U1" )
    if n < 5:
        return codes

    for i in range ( 1, n - 3 ):
        p_im1 = ca_coords[i - 1]
        p_i   = ca_coords[i]
        p_ip1 = ca_coords[i + 1]
        p_ip2 = ca_coords[i + 2]
        p_ip3 = ca_coords[i + 3]

        d2 = float ( np.linalg.norm ( p_im1 - p_ip1 ) )
        d3 = float ( np.linalg.norm ( p_im1 - p_ip2 ) )
        d4 = float ( np.linalg.norm ( p_im1 - p_ip3 ) )

        angle = _angle_between ( p_i - p_im1, p_i - p_ip1 )
        if angle is None:
            continue
        t = np.degrees ( angle )
        a = np.degrees ( _dihedral ( p_im1, p_i, p_ip1, p_ip2 ) )

        is_helix = (
            abs ( t - 89.0 ) <= 12.0 * helix_tolerance_scale
            and abs ( _signed_angle_delta_deg ( a, 50.0 ) ) <= 20.0 * helix_tolerance_scale
            and abs ( d2 - 5.5 ) <= 0.5 * helix_tolerance_scale
            and abs ( d3 - 5.3 ) <= 0.5 * helix_tolerance_scale
            and abs ( d4 - 6.4 ) <= 0.6 * helix_tolerance_scale
        )
        is_strand = (
            abs ( t - 124.0 ) <= 14.0 * strand_tolerance_scale
            and abs ( _signed_angle_delta_deg ( a, -170.0 ) ) <= 45.0 * strand_tolerance_scale
            and abs ( d2 - 6.7 ) <= 0.6 * strand_tolerance_scale
            and abs ( d3 - 9.9 ) <= 0.9 * strand_tolerance_scale
            and abs ( d4 - 12.4 ) <= 1.1 * strand_tolerance_scale
        )

        if is_helix and not is_strand:
            codes[i] = "H"
        elif is_strand and not is_helix:
            codes[i] = "S"

    return _smooth_short_runs ( codes, min_run = min_run )
