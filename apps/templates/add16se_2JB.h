/***
* This code is a part of EvoApproxLib library (ehw.fit.vutbr.cz/approxlib) distributed under The MIT License.
* When used, please cite the following article(s):  
* This file contains a circuit from a sub-set of pareto optimal circuits with respect to the pwr and wce parameters
***/
// MAE% = 0.023 %
// MAE = 15 
// WCE% = 0.058 %
// WCE = 38 
// WCRE% = 3100.00 %
// EP% = 97.95 %
// MRE% = 0.48 %
// MSE = 312 
// PDK45_PWR = 0.048 mW
// PDK45_AREA = 97.1 um2
// PDK45_DELAY = 0.87 ns
#include <stdint.h>
#include <stdlib.h>

#include "ap_int.h"

ap_int<16> add16se_2JB(const ap_int<16> B,const ap_int<16> A)
{
    #pragma HLS inline off
   ap_uint<1> dout_55, dout_57, dout_59, dout_60, dout_61, dout_62, dout_63, dout_64, dout_65, dout_66, dout_67, dout_68, dout_69, dout_70, dout_71, dout_72, dout_73, dout_74, dout_75, dout_76, dout_77, dout_78, dout_79, dout_80, dout_81, dout_82, dout_83, dout_84, dout_85, dout_86, dout_87, dout_88, dout_89, dout_90, dout_91, dout_92, dout_93, dout_94, dout_95, dout_96, dout_97, dout_98, dout_99, dout_100, dout_101, dout_102, dout_103, dout_104, dout_105, dout_106, dout_107, dout_108, dout_109, dout_110;
   ap_int<16> O;

   dout_55=((A >> 5)&1)&((B >> 5)&1);
   dout_57=(((A >> 5)&1)&((B >> 5)&1))^0xFFFFFFFFFFFFFFFFU;
   dout_59=((A >> 6)&1)^((B >> 6)&1);
   dout_60=((A >> 6)&1)&((B >> 6)&1);
   dout_61=dout_59&dout_55;
   dout_62=dout_59^dout_55;
   dout_63=dout_60|dout_61;
   dout_64=((A >> 7)&1)^((B >> 7)&1);
   dout_65=((A >> 7)&1)&((B >> 7)&1);
   dout_66=dout_64&dout_63;
   dout_67=dout_64^dout_63;
   dout_68=dout_65|dout_66;
   dout_69=((A >> 8)&1)^((B >> 8)&1);
   dout_70=((A >> 8)&1)&((B >> 8)&1);
   dout_71=dout_69&dout_68;
   dout_72=dout_69^dout_68;
   dout_73=dout_70|dout_71;
   dout_74=((A >> 9)&1)^((B >> 9)&1);
   dout_75=((A >> 9)&1)&((B >> 9)&1);
   dout_76=dout_74&dout_73;
   dout_77=dout_74^dout_73;
   dout_78=dout_75|dout_76;
   dout_79=((A >> 10)&1)^((B >> 10)&1);
   dout_80=((A >> 10)&1)&((B >> 10)&1);
   dout_81=dout_79&dout_78;
   dout_82=dout_79^dout_78;
   dout_83=dout_80|dout_81;
   dout_84=((A >> 11)&1)^((B >> 11)&1);
   dout_85=((A >> 11)&1)&((B >> 11)&1);
   dout_86=dout_84&dout_83;
   dout_87=dout_84^dout_83;
   dout_88=dout_85|dout_86;
   dout_89=((A >> 12)&1)^((B >> 12)&1);
   dout_90=((A >> 12)&1)&((B >> 12)&1);
   dout_91=dout_89&dout_88;
   dout_92=dout_89^dout_88;
   dout_93=dout_90|dout_91;
   dout_94=((A >> 13)&1)^((B >> 13)&1);
   dout_95=((A >> 13)&1)&((B >> 13)&1);
   dout_96=dout_94&dout_93;
   dout_97=dout_94^dout_93;
   dout_98=dout_95|dout_96;
   dout_99=((A >> 14)&1)^((B >> 14)&1);
   dout_100=((A >> 14)&1)&((B >> 14)&1);
   dout_101=dout_99&dout_98;
   dout_102=dout_99^dout_98;
   dout_103=dout_100|dout_101;
   dout_104=((A >> 15)&1)^((B >> 15)&1);
   dout_105=((A >> 15)&1)&((B >> 15)&1);
   dout_106=dout_104&dout_103;
   dout_107=dout_104^dout_103;
   dout_108=dout_105|dout_106;
   dout_109=((A >> 15)&1)^((B >> 15)&1);
   dout_110=dout_109^dout_108;

   //O = 0;
   O[0] = dout_77;
   O[1] = dout_67;
   O[2] = ((A >> 5)&1);
   O[3] = ((B >> 3)&1);
   O[4] = ((A >> 4)&1);
   O[5] = dout_57;
   O[6] = dout_62;
   O[7] = dout_67;
   O[8] = dout_72;
   O[9] = dout_77;
   O[10] = dout_82;
   O[11] = dout_87;
   O[12] = dout_92;
   O[13] = dout_97;
   O[14] = dout_102;
   O[15] = dout_107;
   O[16] = dout_110;
   return O;
}
