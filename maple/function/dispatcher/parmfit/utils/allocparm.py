class ForceFieldDB:
    residue_templates   # 标准残基 / cap 的原子名、类型、默认电荷、连接
    bond_params
    angle_params
    dihedral_params
    improper_params
    vdw_params
    cmap_tables         # ff19SB 才用