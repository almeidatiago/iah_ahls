#include "comp/mul16s_HEB.h"
#include "comp/mul16s_HFZ.h"
#include "comp/mul16s_GK2.h"
#include "comp/mul16s_GAT.h"
#include "comp/mul16s_HDG.h"
#include "comp/mul16s_G80.h"
#include "comp/mul16s_G7F.h"
#include "comp/mul16s_G7Z.h"

#include "comp/add16se_RCA.h"
#include "comp/add16se_2GE.h"
#include "comp/add16se_25S.h"
#include "comp/add16se_2AS.h"
#include "comp/add16se_2JB.h"
#include "comp/add16se_294.h"
#include "comp/add16se_2JY.h"
#include "comp/add16se_1Y7.h"
#include "comp/add16se_259.h"
#include "comp/add16se_26Q.h"
#include "comp/add16se_29A.h"
#include "comp/add16se_2E1.h"


int32_t (*mul[8]) (int16_t, int16_t) = {mul16s_HEB, mul16s_HFZ, mul16s_GK2, mul16s_GAT, mul16s_HDG, mul16s_G80, mul16s_G7F, mul16s_G7Z};
int16_t (*add[12]) (int16_t, int16_t) = {add16se_RCA, add16se_2GE, add16se_25S, add16se_2AS, add16se_2JB, add16se_294, add16se_2JY, add16se_1Y7, add16se_259, add16se_26Q, add16se_29A, add16se_2E1};
