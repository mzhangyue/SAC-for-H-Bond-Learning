/**-----------------------------------------------------------------------------                                                                                                  
**                                                                                                                                                                               
**  Copyright (C) : Structural Bioinformatics Laboratory, Boston University.                                                                                                                        
**                                                                                                                                                                               
**  This software was developed at the Boston University 2006-2011, by                                                                                                      
**  Structural Bioinformatics Laboratory, as part of NIH funded research.                                                                                                                      
**                                                                                                                                                                               
**  Explicit permission is hereby granted to US Universities and US                                                                                                     
**  Government supported Research Institutions to copy and modify this                                                                                                           
**  software for educational and research purposes, provided copies include                                                                                                      
**  this notice. This software (or modified copies thereof) may not be                                                                                                           
**  distributed to any other institution without express permission from the                                                                                                     
**  Structural Bioinformatics Laboratory and  Boston University. Requests to use this software (or                                                                                 **  modified copies therof) in any other way should be sent to Dima Kozakov,                                                                                                     
**  Department of Biomedical Engineering: "midas@bu.edu".                                                                                                                  
**                                                                                                                                                                               
**---------------------------------------------------------------------------*/
#ifndef _MOL_H_
#define _MOL_H_

#ifndef _PRINT_DEPRECATED_
#define _PRINT_DEPRECATED_ fprintf (stderr, "%s: Deprecated function\n", __func__);
#endif
#ifndef _mol_error
#define _mol_error(format,...) fprintf (stderr, "%s in %s@%d: " format "\n", __func__, __FILE__, __LINE__, __VA_ARGS__)
#endif
#ifndef strequal
#define strequal(s1,s2) (!strcmp(s1,s2))
#endif
#ifndef strnequal
#define strnequal(s1,s2,n) (!strncmp(s1,s2,n))
#endif

typedef unsigned int uint;

#include <fftw3.h>
#include "mol.0.0.6/mem.h"
#include "mol.0.0.6/myhelpers.h"
#include "mol.0.0.6/prms.h"
#include "mol.0.0.6/io.h"
#include "mol.0.0.6/icharmm.h"
#include "mol.0.0.6/mol2.h"
#include "mol.0.0.6/bond.h"
#include "mol.0.0.6/atom.h"
#include "mol.0.0.6/atom_group.h"
#include "mol.0.0.6/_atom_group_copy_from_deprecated.h"
#include "mol.0.0.6/xyz.h"
#include "mol.0.0.6/init.h"
#include "mol.0.0.6/tvector.h"
#include "mol.0.0.6/move.h"
#include "mol.0.0.6/protein.h"
#include "mol.0.0.6/phys.h"
#include "mol.0.0.6/pdb.h"
#include "mol.0.0.6/ms.h"
#include "mol.0.0.6/octree.h"
#include "mol.0.0.6/matrix.h"
#include "mol.0.0.6/sasa.h"
#include "mol.0.0.6/potential.h"
#include "mol.0.0.6/energy.h"
#include "mol.0.0.6/benergy.h"
#include "mol.0.0.6/nbenergy.h"
#include "mol.0.0.6/mask.h"
#include "mol.0.0.6/minimize.h"
#include "mol.0.0.6/compare.h"
#include "mol.0.0.6/gbsa.h"
#include "mol.0.0.6/subag.h"
#include "mol.0.0.6/hbond.h"
#include "mol.0.0.6/rotamers.h"
#include "mol.0.0.6/quaternions.h"
#include "mol.0.0.6/version.h"
#include "mol.0.0.6/run_octree.h"
#endif
