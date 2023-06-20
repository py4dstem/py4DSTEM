# single files
file_ids = {
    'sample_diffraction_pattern' : (
        'a_diffraction_pattern.h5',
        '1ymYMnuDC0KV6dqduxe2O1qafgSd0jjnU',
    ),
    'Au_sim' : (
        'au_sim.h5',
        '1PmbCYosA1eYydWmmZebvf6uon9k_5g_S',
    ),
    'carbon_nanotube' : (
        'carbon_nanotube.h5',
        '1bHv3u61Cr-y_GkdWHrJGh1lw2VKmt3UM',
    ),
    'Si_SiGe_exp' : (
        'Si_SiGe_exp.h5',
        '1fXNYSGpe6w6E9RBA-Ai_owZwoj3w8PNC',
    ),
    'Si_SiGe_probe' : (
        'Si_SiGe_probe.h5',
        '141Tv0YF7c5a-MCrh3CkY_w4FgWtBih80',
    ),
    'Si_SiGe_EELS_strain' : (
        'Si_SiGe_EELS_strain.h5',
        '1klkecq8IuEOYB-bXchO7RqOcgCl4bmDJ',
    ),
    'AuAgPd_wire' : (
        'AuAgPd_wire.h5',
        '1OQYW0H6VELsmnLTcwicP88vo2V5E3Oyt',
    ),
    'AuAgPd_wire_probe' : (
        'AuAgPd_wire_probe.h5',
        '17OduUKpxVBDumSK_VHtnc2XKkaFVN8kq',
    ),
    'polycrystal_2D_WS2' : (
        'polycrystal_2D_WS2.h5',
        '1AWB3-UTPiTR9dgrEkNFD7EJYsKnbEy0y',
    ),
    'WS2cif' : (
        'WS2.cif',
        '13zBl6aFExtsz_sew-L0-_ALYJfcgHKjo',
    ),
    'polymers' : (
        'polymers.h5',
        '1lK-TAMXN1MpWG0Q3_4vss_uEZgW2_Xh7',
    ),
    'vac_probe' : (
        'vac_probe.h5',
        '1QTcSKzZjHZd1fDimSI_q9_WsAU25NIXe',
    ),
    'small_dm3' : (
        'small_dm3.dm3',
        '1RxI1QY6vYMDqqMVPt5GBN6Q_iCwHFU4B'
    ),
    'FCU-Net' : (
        'filename.name',
        '1-KX0saEYfhZ9IJAOwabH38PCVtfXidJi',
    ),
}

# collections of files
collection_ids = {
    'tutorials' : (
        'Au_sim',
        'carbon_nanotube',
        'Si_SiGe_exp',
        'Si_SiGe_probe',
        'Si_SiGe_EELS_strain',
        'AuAgPd_wire',
        'AuAgPd_wire_probe',
        'polycrystal_2D_WS2',
        'WS2cif',
        'polymers',
        'vac_probe',
    ),
    'io_test_data' : (
        'small_dm3',
        'vac_probe',
    ),
}


def get_sample_file_ids():
    return {
        'files' : file_ids.keys(),
        'collections' : collection_ids.keys()
    }


