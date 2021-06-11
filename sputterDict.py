data = {
        "Mg2SiO4"   : {"u0": 5.7, "md": 20.0, "zd": 10.0, "K": 0.1 , "rhod": 3.20, "a0": 2.055},
        "Al2O3"     : {"u0": 8.5, "md": 20.4, "zd": 10.0, "K": 0.08, "rhod": 3.95, "a0": 1.718},
        "SiC"       : {"u0": 6.3, "md": 20.0, "zd": 10.0,  "K": 0.3, "rhod": 3.21, "a0": 1.702},
        "C" : {"u0": 4, "zd": 6, "md": 12, "K":0.61,"rhod":2.26, "a0": 1.281},
        "Si" : {"u0": 4.66, "zd": 14, "md": 28, "K":0.43,"rhod":2.32, "a0": 1.684},
        "SiO2" : {"u0": 6.42, "zd": 10, "md": 20, "K":0.1,"rhod":2.64, "a0": 2.080},
        "MgO" : {"u0": 5.17, "zd": 10, "md": 20, "K":0.06,"rhod":3.56, "a0": 1.646},
        "MgSiO3" : {"u0": 6, "zd": 10, "md": 20, "K":0.1,"rhod":3.18, "a0": 2.319},
        "Fe" : {"u0": 4.31, "zd": 26, "md": 56, "K":0.23,"rhod":7.89, "a0": 1.411},
        "FeS" : {"u0": 4.12, "zd": 21, "md": 44, "K":0.18,"rhod":4.84, "a0": 1.932},
        "Mg2SiO4-Mg" : {"u0": 5.8, "zd": 10, "md": 20, "K":0.1,"rhod":3.2, "a0": 2.055},
        "Mg2SiO4-SiO" : {"u0": 5.8, "zd": 10, "md": 20, "K":0.1,"rhod":3.2, "a0": 2.589},
        "Fe3O4" : {"u0": 4.98, "zd": 15.7, "md": 33.1, "K":0.15,"rhod":5.21, "a0": 1.805},
        #"FeO" : {"u0": , "zd": 17, "md": 35.9, "K":,"rhod":5.745, "a0": 1.682},
        "SiC" : {"u0": 6.3, "zd": 10, "md": 20, "K":0.3,"rhod":3.16, "a0": 1.702},
        #"Ti" : {"u0": , "zd": 22, "md": 47.867, "K":,"rhod":4.507, "a0": 1.689},
        #"V" : {"u0": , "zd": 23, "md": 50.9415, "K":,"rhod":6.11, "a0": 1.490},
        #"Cr" : {"u0": , "zd": 24, "md": 51.9961, "K":,"rhod":7.19, "a0": 1.421},
        #"Co" : {"u0": , "zd": 27, "md": 58.9332, "K":,"rhod":8.9, "a0": 1.383},
        #"Ni" : {"u0": , "zd": 28, "md": 58.6934, "K":,"rhod":8.908, "a0": 1.377},
        #"Cu" : {"u0": , "zd": 29, "md": 63.546, "K":,"rhod":8.96, "a0": 1.412},
        #"TiC" : {"u0": , "zd": 14, "md": 29.94, "K":,"rhod":4.93, "a0": 1.689}
}


ions = {
    "Ag":	{"mi":	107.868,	"zi":	47.},
    "Al":	{"mi":	26.9815,	"zi":	13.},
    "Ar":	{"mi":	39.948,	"zi":	19.},
    "As":	{"mi":	74.9216,	"zi":	33.},
    "Au":	{"mi":	196.9665,	"zi":	79.},
    "B":	{"mi":	10.811	,	"zi":	5.},
    "Ba":	{"mi":	137.33,	"zi":	56.},
    "Be":	{"mi":	9.0122,	"zi":	4.},
    "Bi":	{"mi":	208.9804,	"zi":	83.},
    "Br":	{"mi":	79.904,	"zi":	35.},
    "C":	{"mi":	12.0107,	"zi":	6.},
    "Ca":	{"mi":	40.078,	"zi":	20.},
    "Cd":	{"mi":	112.41	,	"zi":	48.},
    "Ce":	{"mi":	140.12,	"zi":	58.},
    "Cl":	{"mi":	35.453,	"zi":	17.},
    "Co":	{"mi":	58.9332,	"zi":	28.},
    "Cr":	{"mi":	51.9961,	"zi":	24.},
    "Cs":	{"mi":	132.9054,	"zi":	55.},
    "Cu":	{"mi":	63.546,	"zi":	29.},
    "Dy":	{"mi":	162.5	,	"zi":	66.},
    "Er":	{"mi":	167.26,	"zi":	68.},
    "Eu":	{"mi":	151.96,	"zi":	63.},
    "F":	{"mi":	18.9984,	"zi":	9.},
    "Fe":	{"mi":	55.845,	"zi":	26.},
    "Ga":	{"mi":	69.723,	"zi":	31.},
    "Gd":	{"mi":	157.25,	"zi":	64.},
    "Ge":	{"mi":	72.64	,	"zi":	32.},
    "H":	{"mi":	1.0079,	"zi":	1.},
    "He":	{"mi":	4.0026,	"zi":	2.},
    "Hf":	{"mi":	178.49,	"zi":	72.},
    "Hg":	{"mi":	200.59,	"zi":	80.},
    "Ho":	{"mi":	164.9304,	"zi":	67.},
    "I":	{"mi":	126.9045,	"zi":	53.},
    "In":	{"mi":	114.82	,	"zi":	49.},
    "Ir":	{"mi":	192.22,	"zi":	77.},
    "K":	{"mi":	39.0983,	"zi":	18.},
    "Kr":	{"mi":	83.8	,	"zi":	36.},
    "La":	{"mi":	138.9055,	"zi":	57.},
    "Li":	{"mi":	6.941	,	"zi":	3.},
    "Lu":	{"mi":	174.967,	"zi":	71.},
    "Mg":	{"mi":	24.305,	"zi":	12.},
    "Mn":	{"mi":	54.938,	"zi":	25.},
    "Mo":	{"mi":	95.94,		"zi":	42.},
    "N":	{"mi":	14.0067,	"zi":	7.},
    "Na":	{"mi":	22.9897,	"zi":	11.},
    "Nb":	{"mi":	92.9064,	"zi":	41.},
    "Nd":	{"mi":	144.24,	"zi":	60.},
    "Ne":	{"mi":	20.1797,	"zi":	10.},
    "Ni":	{"mi":	58.6934,	"zi":	27.},
    "O":	{"mi":	15.9994,	"zi":	8.},
    "Os":	{"mi":	190.2	,	"zi":	76.},
    "P":	{"mi":	30.9738,	"zi":	15.},
    "Pb":	{"mi":	207.2	,	"zi":	82.},
    "Pd":	{"mi":	106.4	,	"zi":	46.},
    "Pm":	{"mi":	145	,	"zi":	61.},
    "Po":	{"mi":	209,		"zi":	84.},
    "Pr":	{"mi":	140.9077,	"zi":	59.},
    "Pt":	{"mi":	195.09,	"zi":	78.},
    "Rb":	{"mi":	85.4678,	"zi":	37.},
    "Re":	{"mi":	186.207,	"zi":	75.},
    "Rh":	{"mi":	102.9055,	"zi":	45.},
    "Ru":	{"mi":	101.07,	"zi":	44.},
    "S":	{"mi":	32.065,	"zi":	16.},
    "Sb":	{"mi":	121.75,	"zi":	51.},
    "Sc":	{"mi":	44.9559,	"zi":	21.},
    "Se":	{"mi":	78.96	,	"zi":	34.},
    "Si":	{"mi":	28.0855,	"zi":	14.},
    "Sm":	{"mi":	150.4	,	"zi":	62.},
    "Sn":	{"mi":	118.69	,	"zi":	50.},
    "Sr":	{"mi":	87.62	,	"zi":	38.},
    "Ta":	{"mi":	180.9479,	"zi":	73.},
    "Tb":	{"mi":	158.9254,	"zi":	65.},
    "Tc":	{"mi":	98	,	"zi":	43.},
    "Te":	{"mi":	127.6	,	"zi":	52.},
    "Ti":	{"mi":	47.867,	"zi":	22.},
    "Tl":	{"mi":	204.37,	"zi":	81.},
    "Tm":	{"mi":	168.9342,	"zi":	69.},
    "V":	{"mi":	50.9415,	"zi":	23.},
    "W":	{"mi":	183.85,	"zi":	74.},
    "Xe":	{"mi":	131.3	,	"zi":	54.},
    "Y":	{"mi":	88.9059,	"zi":	39.},
    "Yb":	{"mi":	173.04,	"zi":	70.},
    "Zn":	{"mi":	65.39	,	"zi":	30.},
    "Zr":	{"mi":	91.22	,	"zi":	40.},
    "SiO":       {"mi":  44.0849,        "zi":   22.}
}

"""
ions = {
        "h+"    : { "mi": 1.004,  "zi": 1.0},
        "o+"    : { "mi": 15.999, "zi": 8.0},
        "mg+"   : { "mi": 24.305, "zi": 12.0},
        "he+"   : { "mi": 4.0026, "zi": 2.0},
        "ar+"   : { "mi": 39.948, "zi": 18.0}
}
ionALL = {
        "H":   {"mi":	1.0079, "zi":	1},
        "He":   {"mi":	4.0026, "zi":	2},
        "Li":   {"mi":	6.941, "zi":	3},
        "Be":   {"mi":	9.0122, "zi":	4},
        "B":    {"mi":	10.811, "zi":	5},
        "C":    {"mi":	12.0107, "zi":	6},
        "N":    {"mi":	14.0067, "zi":	7},
        "O":    {"mi":	15.9994, "zi":	8},
        "F":    {"mi":	18.9984, "zi":	9},
        "Ne":   {"mi":	20.1797, "zi":	10},
        "Na":   {"mi":	22.9897, "zi":	11},
        "Mg":   { "mi":	24.305, "zi":	12},
        "Al":   {"mi":	26.9815, "zi":	13},
        "Si":   {"mi":	28.0855, "zi":	14},
        "P":    {"mi":	30.9738, "zi":	15},
        "S":    {"mi":	32.065, "zi":	16},
        "Cl":   {"mi":	35.453, "zi":	17},
        "K":    {"mi":	39.0983, "zi":	18},
        "Ar":   {"mi":	39.948, "zi":	19},
        "Ca":   {"mi":	40.078, "zi":	20},
        "SiO":  {"mi":	44.0849, "zi":	22},
        "Sc":   {"mi":	44.9559, "zi":	21},
        "Ti":   {"mi":	47.867, "zi":	22},
        "V":    {"mi":	50.9415, "zi":	23},
        "Cr":   {"mi":	51.9961, "zi":	24},
        "Mn":   {"mi":	54.938, "zi":	25},
        "Fe":   {"mi":	55.845, "zi":	26},
        "Ni":   {"mi":	58.6934, "zi":	27},
        "Co":   {"mi":	58.9332, "zi":	28},
        "Cu":   {"mi":	63.546, "zi":	29},
        "Zn":   {"mi":	65.39, "zi":	30},
        "Ga":   {"mi":	69.723, "zi":	31},
        "Ge":   {"mi":	72.64, "zi":	32},
        "As":   {"mi":	74.9216, "zi":	33},
        "Se":   {"mi":	78.96, "zi":	34},
        "Br":   {"mi":	79.904, "zi":	35},
        "Kr":   {"mi":	83.8, "zi":	36},
        "Rb":   {"mi":	85.4678, "zi":	37},
        "Sr":   {"mi":	87.62, "zi":	38},
        "Y":    {"mi":	88.9059, "zi":	39},
        "Zr":   {"mi":	91.22, "zi":	40},
        "Nb":   {"mi":	92.9064, "zi":	41},
        "Mo":   {"mi":	95.94, "zi":	42},
        "Tc":   {"mi":	98, "zi":	43},
        "Ru":   {"mi":	101.07, "zi":	44},
        "Rh":   {"mi":	102.9055, "zi":	45},
        "Pd":   {"mi":	106.4, "zi":	46},
        "Ag":   {"mi":	107.868, "zi":	47},
        "Cd":   {"mi":	112.41, "zi":	48},
        "In":   {"mi":	114.82, "zi":	49},
        "Sn":   {"mi":	118.69, "zi":	50},
        "Sb":   {"mi":	121.75, "zi":	51},
        "I":    {"mi":	126.9045, "zi":	53},
        "Te":   {"mi":	127.6, "zi":	52},
        "Xe":   {"mi":	131.3, "zi":	54},
        "Cs":   {"mi":	132.9054, "zi":	55},
        "Ba":   {"mi":	137.33, "zi":	56},
        "La":   {"mi":	138.9055, "zi":	57},
        "Ce":   {"mi":	140.12, "zi":	58},
        "Pr":   {"mi":	140.9077, "zi":	59},
        "Nd": {"mi":	144.24, "zi":	60},
        "Pm": {"mi":	145, "zi":	61},
        "Sm": {"mi":	150.4, "zi":	62},
        "Eu": {"mi":	151.96, "zi":	63},
        "Gd": {"mi":	157.25, "zi":	64},
        "Tb": {"mi":	158.9254, "zi":	65},
        "Dy": {"mi":	162.5, "zi":	66},
        "Ho": {"mi":	164.9304, "zi":	67},
        "Er": {"mi":	167.26, "zi":	68},
        "Tm": {"mi":	168.9342, "zi":	69},
        "Yb": {"mi":	173.04, "zi":	70},
        "Lu": {"mi":	174.967, "zi":	71},
        "Hf": {"mi":	178.49, "zi":	72},
        "Ta": {"mi":	180.9479, "zi":	73},
        "W": {"mi":	183.85, "zi":	74},
        "Re": {"mi":	186.207, "zi":	75},
        "Os": {"mi":	190.2, "zi":	76},
        "Ir": {"mi":	192.22, "zi":	77},
        "Pt": {"mi":	195.09, "zi":	78},
        "Au": {"mi":	196.9665, "zi":	79},
        "Hg": {"mi":	200.59, "zi":	80},
        "Tl": {"mi":	204.37, "zi":	81},
        "Pb": {"mi":	207.2, "zi":	82},
        "Bi": {"mi":	208.9804, "zi":	83},
        "Po": {"mi":	209, "zi":	84},
        "At": {"mi":	210, "zi":	85},
        "Rn": {"mi":	222, "zi":	86},
        "Fr": {"mi":	223, "zi":	87},
        "Ra": {"mi":	226.0254, "zi":	88},
        "Ac": {"mi":	227.0278, "zi":	89},
        "Pa	": {"mi":	231.0359, "zi":	91},
        "Th	": {"mi":	232.0381, "zi":	90},
        "Np	": {"mi":	237.0482, "zi":	93},
        "U	": {"mi":	238.0289, "zi":	92},
        "Am	": {"mi":	243, "zi":	95},
        "Pu	": {"mi":	244, "zi":	94},
        "Cm	": {"mi":	247, "zi":	96},
        "Bk	": {"mi":	247, "zi":	97},
        "Cf	": {"mi":	251, "zi":	98},
        "Es	": {"mi":	252, "zi":	99},
        "Fm	": {"mi":	257, "zi":	100},
        "Md	": {"mi":	258, "zi":	101},
        "No	": {"mi":	259, "zi":	102},
        "Rf	": {"mi":	261, "zi":	104},
        "Lr	": {"mi":	262, "zi":	103},
        "Db	": {"mi":	262, "zi":	105},
        "Bh	": {"mi":	264, "zi":	107},
        "Sg	": {"mi":	266, "zi":	106},
        "Mt	": {"mi":	268, "zi":	109},
        "Rg	": {"mi":	272, "zi":	111},
        "Hs	": {"mi":	277, "zi":	108}
}
"""
