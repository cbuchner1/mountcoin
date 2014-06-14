typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define AES_WITHOUT_SHARED 0

// 0 is good for SM 35
// 1 is good for SM 30
#define DATA_LAYOUT_SWITCH 0

// aus heavy.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

uint32_t *d_keccakOutputData[8];
uint32_t *d_keccakOutputExpandedAESKey1[8];
uint32_t *d_keccakOutputExpandedAESKey2[8];
uint32_t *d_hashOutput[8];
uint32_t *d_prepMemory[8];
uint32_t *d_abInput[8];

#define SPH_C64(x)    ((uint64_t)(x ## ULL))
#define SPH_C32(x)    ((uint32_t)(x ## U))

#include "cuda_helper.h"
////////////// TABLES FROM SPH

/* AES Helper for inline-usage from SPH */
#define AESx(x) SPH_C32(x)

static const uint32_t h_AES0[256] = {
	AESx(0xA56363C6), AESx(0x847C7CF8), AESx(0x997777EE), AESx(0x8D7B7BF6),
	AESx(0x0DF2F2FF), AESx(0xBD6B6BD6), AESx(0xB16F6FDE), AESx(0x54C5C591),
	AESx(0x50303060), AESx(0x03010102), AESx(0xA96767CE), AESx(0x7D2B2B56),
	AESx(0x19FEFEE7), AESx(0x62D7D7B5), AESx(0xE6ABAB4D), AESx(0x9A7676EC),
	AESx(0x45CACA8F), AESx(0x9D82821F), AESx(0x40C9C989), AESx(0x877D7DFA),
	AESx(0x15FAFAEF), AESx(0xEB5959B2), AESx(0xC947478E), AESx(0x0BF0F0FB),
	AESx(0xECADAD41), AESx(0x67D4D4B3), AESx(0xFDA2A25F), AESx(0xEAAFAF45),
	AESx(0xBF9C9C23), AESx(0xF7A4A453), AESx(0x967272E4), AESx(0x5BC0C09B),
	AESx(0xC2B7B775), AESx(0x1CFDFDE1), AESx(0xAE93933D), AESx(0x6A26264C),
	AESx(0x5A36366C), AESx(0x413F3F7E), AESx(0x02F7F7F5), AESx(0x4FCCCC83),
	AESx(0x5C343468), AESx(0xF4A5A551), AESx(0x34E5E5D1), AESx(0x08F1F1F9),
	AESx(0x937171E2), AESx(0x73D8D8AB), AESx(0x53313162), AESx(0x3F15152A),
	AESx(0x0C040408), AESx(0x52C7C795), AESx(0x65232346), AESx(0x5EC3C39D),
	AESx(0x28181830), AESx(0xA1969637), AESx(0x0F05050A), AESx(0xB59A9A2F),
	AESx(0x0907070E), AESx(0x36121224), AESx(0x9B80801B), AESx(0x3DE2E2DF),
	AESx(0x26EBEBCD), AESx(0x6927274E), AESx(0xCDB2B27F), AESx(0x9F7575EA),
	AESx(0x1B090912), AESx(0x9E83831D), AESx(0x742C2C58), AESx(0x2E1A1A34),
	AESx(0x2D1B1B36), AESx(0xB26E6EDC), AESx(0xEE5A5AB4), AESx(0xFBA0A05B),
	AESx(0xF65252A4), AESx(0x4D3B3B76), AESx(0x61D6D6B7), AESx(0xCEB3B37D),
	AESx(0x7B292952), AESx(0x3EE3E3DD), AESx(0x712F2F5E), AESx(0x97848413),
	AESx(0xF55353A6), AESx(0x68D1D1B9), AESx(0x00000000), AESx(0x2CEDEDC1),
	AESx(0x60202040), AESx(0x1FFCFCE3), AESx(0xC8B1B179), AESx(0xED5B5BB6),
	AESx(0xBE6A6AD4), AESx(0x46CBCB8D), AESx(0xD9BEBE67), AESx(0x4B393972),
	AESx(0xDE4A4A94), AESx(0xD44C4C98), AESx(0xE85858B0), AESx(0x4ACFCF85),
	AESx(0x6BD0D0BB), AESx(0x2AEFEFC5), AESx(0xE5AAAA4F), AESx(0x16FBFBED),
	AESx(0xC5434386), AESx(0xD74D4D9A), AESx(0x55333366), AESx(0x94858511),
	AESx(0xCF45458A), AESx(0x10F9F9E9), AESx(0x06020204), AESx(0x817F7FFE),
	AESx(0xF05050A0), AESx(0x443C3C78), AESx(0xBA9F9F25), AESx(0xE3A8A84B),
	AESx(0xF35151A2), AESx(0xFEA3A35D), AESx(0xC0404080), AESx(0x8A8F8F05),
	AESx(0xAD92923F), AESx(0xBC9D9D21), AESx(0x48383870), AESx(0x04F5F5F1),
	AESx(0xDFBCBC63), AESx(0xC1B6B677), AESx(0x75DADAAF), AESx(0x63212142),
	AESx(0x30101020), AESx(0x1AFFFFE5), AESx(0x0EF3F3FD), AESx(0x6DD2D2BF),
	AESx(0x4CCDCD81), AESx(0x140C0C18), AESx(0x35131326), AESx(0x2FECECC3),
	AESx(0xE15F5FBE), AESx(0xA2979735), AESx(0xCC444488), AESx(0x3917172E),
	AESx(0x57C4C493), AESx(0xF2A7A755), AESx(0x827E7EFC), AESx(0x473D3D7A),
	AESx(0xAC6464C8), AESx(0xE75D5DBA), AESx(0x2B191932), AESx(0x957373E6),
	AESx(0xA06060C0), AESx(0x98818119), AESx(0xD14F4F9E), AESx(0x7FDCDCA3),
	AESx(0x66222244), AESx(0x7E2A2A54), AESx(0xAB90903B), AESx(0x8388880B),
	AESx(0xCA46468C), AESx(0x29EEEEC7), AESx(0xD3B8B86B), AESx(0x3C141428),
	AESx(0x79DEDEA7), AESx(0xE25E5EBC), AESx(0x1D0B0B16), AESx(0x76DBDBAD),
	AESx(0x3BE0E0DB), AESx(0x56323264), AESx(0x4E3A3A74), AESx(0x1E0A0A14),
	AESx(0xDB494992), AESx(0x0A06060C), AESx(0x6C242448), AESx(0xE45C5CB8),
	AESx(0x5DC2C29F), AESx(0x6ED3D3BD), AESx(0xEFACAC43), AESx(0xA66262C4),
	AESx(0xA8919139), AESx(0xA4959531), AESx(0x37E4E4D3), AESx(0x8B7979F2),
	AESx(0x32E7E7D5), AESx(0x43C8C88B), AESx(0x5937376E), AESx(0xB76D6DDA),
	AESx(0x8C8D8D01), AESx(0x64D5D5B1), AESx(0xD24E4E9C), AESx(0xE0A9A949),
	AESx(0xB46C6CD8), AESx(0xFA5656AC), AESx(0x07F4F4F3), AESx(0x25EAEACF),
	AESx(0xAF6565CA), AESx(0x8E7A7AF4), AESx(0xE9AEAE47), AESx(0x18080810),
	AESx(0xD5BABA6F), AESx(0x887878F0), AESx(0x6F25254A), AESx(0x722E2E5C),
	AESx(0x241C1C38), AESx(0xF1A6A657), AESx(0xC7B4B473), AESx(0x51C6C697),
	AESx(0x23E8E8CB), AESx(0x7CDDDDA1), AESx(0x9C7474E8), AESx(0x211F1F3E),
	AESx(0xDD4B4B96), AESx(0xDCBDBD61), AESx(0x868B8B0D), AESx(0x858A8A0F),
	AESx(0x907070E0), AESx(0x423E3E7C), AESx(0xC4B5B571), AESx(0xAA6666CC),
	AESx(0xD8484890), AESx(0x05030306), AESx(0x01F6F6F7), AESx(0x120E0E1C),
	AESx(0xA36161C2), AESx(0x5F35356A), AESx(0xF95757AE), AESx(0xD0B9B969),
	AESx(0x91868617), AESx(0x58C1C199), AESx(0x271D1D3A), AESx(0xB99E9E27),
	AESx(0x38E1E1D9), AESx(0x13F8F8EB), AESx(0xB398982B), AESx(0x33111122),
	AESx(0xBB6969D2), AESx(0x70D9D9A9), AESx(0x898E8E07), AESx(0xA7949433),
	AESx(0xB69B9B2D), AESx(0x221E1E3C), AESx(0x92878715), AESx(0x20E9E9C9),
	AESx(0x49CECE87), AESx(0xFF5555AA), AESx(0x78282850), AESx(0x7ADFDFA5),
	AESx(0x8F8C8C03), AESx(0xF8A1A159), AESx(0x80898909), AESx(0x170D0D1A),
	AESx(0xDABFBF65), AESx(0x31E6E6D7), AESx(0xC6424284), AESx(0xB86868D0),
	AESx(0xC3414182), AESx(0xB0999929), AESx(0x772D2D5A), AESx(0x110F0F1E),
	AESx(0xCBB0B07B), AESx(0xFC5454A8), AESx(0xD6BBBB6D), AESx(0x3A16162C)
};

static const uint32_t h_AES1[256] = {
	AESx(0x6363C6A5), AESx(0x7C7CF884), AESx(0x7777EE99), AESx(0x7B7BF68D),
	AESx(0xF2F2FF0D), AESx(0x6B6BD6BD), AESx(0x6F6FDEB1), AESx(0xC5C59154),
	AESx(0x30306050), AESx(0x01010203), AESx(0x6767CEA9), AESx(0x2B2B567D),
	AESx(0xFEFEE719), AESx(0xD7D7B562), AESx(0xABAB4DE6), AESx(0x7676EC9A),
	AESx(0xCACA8F45), AESx(0x82821F9D), AESx(0xC9C98940), AESx(0x7D7DFA87),
	AESx(0xFAFAEF15), AESx(0x5959B2EB), AESx(0x47478EC9), AESx(0xF0F0FB0B),
	AESx(0xADAD41EC), AESx(0xD4D4B367), AESx(0xA2A25FFD), AESx(0xAFAF45EA),
	AESx(0x9C9C23BF), AESx(0xA4A453F7), AESx(0x7272E496), AESx(0xC0C09B5B),
	AESx(0xB7B775C2), AESx(0xFDFDE11C), AESx(0x93933DAE), AESx(0x26264C6A),
	AESx(0x36366C5A), AESx(0x3F3F7E41), AESx(0xF7F7F502), AESx(0xCCCC834F),
	AESx(0x3434685C), AESx(0xA5A551F4), AESx(0xE5E5D134), AESx(0xF1F1F908),
	AESx(0x7171E293), AESx(0xD8D8AB73), AESx(0x31316253), AESx(0x15152A3F),
	AESx(0x0404080C), AESx(0xC7C79552), AESx(0x23234665), AESx(0xC3C39D5E),
	AESx(0x18183028), AESx(0x969637A1), AESx(0x05050A0F), AESx(0x9A9A2FB5),
	AESx(0x07070E09), AESx(0x12122436), AESx(0x80801B9B), AESx(0xE2E2DF3D),
	AESx(0xEBEBCD26), AESx(0x27274E69), AESx(0xB2B27FCD), AESx(0x7575EA9F),
	AESx(0x0909121B), AESx(0x83831D9E), AESx(0x2C2C5874), AESx(0x1A1A342E),
	AESx(0x1B1B362D), AESx(0x6E6EDCB2), AESx(0x5A5AB4EE), AESx(0xA0A05BFB),
	AESx(0x5252A4F6), AESx(0x3B3B764D), AESx(0xD6D6B761), AESx(0xB3B37DCE),
	AESx(0x2929527B), AESx(0xE3E3DD3E), AESx(0x2F2F5E71), AESx(0x84841397),
	AESx(0x5353A6F5), AESx(0xD1D1B968), AESx(0x00000000), AESx(0xEDEDC12C),
	AESx(0x20204060), AESx(0xFCFCE31F), AESx(0xB1B179C8), AESx(0x5B5BB6ED),
	AESx(0x6A6AD4BE), AESx(0xCBCB8D46), AESx(0xBEBE67D9), AESx(0x3939724B),
	AESx(0x4A4A94DE), AESx(0x4C4C98D4), AESx(0x5858B0E8), AESx(0xCFCF854A),
	AESx(0xD0D0BB6B), AESx(0xEFEFC52A), AESx(0xAAAA4FE5), AESx(0xFBFBED16),
	AESx(0x434386C5), AESx(0x4D4D9AD7), AESx(0x33336655), AESx(0x85851194),
	AESx(0x45458ACF), AESx(0xF9F9E910), AESx(0x02020406), AESx(0x7F7FFE81),
	AESx(0x5050A0F0), AESx(0x3C3C7844), AESx(0x9F9F25BA), AESx(0xA8A84BE3),
	AESx(0x5151A2F3), AESx(0xA3A35DFE), AESx(0x404080C0), AESx(0x8F8F058A),
	AESx(0x92923FAD), AESx(0x9D9D21BC), AESx(0x38387048), AESx(0xF5F5F104),
	AESx(0xBCBC63DF), AESx(0xB6B677C1), AESx(0xDADAAF75), AESx(0x21214263),
	AESx(0x10102030), AESx(0xFFFFE51A), AESx(0xF3F3FD0E), AESx(0xD2D2BF6D),
	AESx(0xCDCD814C), AESx(0x0C0C1814), AESx(0x13132635), AESx(0xECECC32F),
	AESx(0x5F5FBEE1), AESx(0x979735A2), AESx(0x444488CC), AESx(0x17172E39),
	AESx(0xC4C49357), AESx(0xA7A755F2), AESx(0x7E7EFC82), AESx(0x3D3D7A47),
	AESx(0x6464C8AC), AESx(0x5D5DBAE7), AESx(0x1919322B), AESx(0x7373E695),
	AESx(0x6060C0A0), AESx(0x81811998), AESx(0x4F4F9ED1), AESx(0xDCDCA37F),
	AESx(0x22224466), AESx(0x2A2A547E), AESx(0x90903BAB), AESx(0x88880B83),
	AESx(0x46468CCA), AESx(0xEEEEC729), AESx(0xB8B86BD3), AESx(0x1414283C),
	AESx(0xDEDEA779), AESx(0x5E5EBCE2), AESx(0x0B0B161D), AESx(0xDBDBAD76),
	AESx(0xE0E0DB3B), AESx(0x32326456), AESx(0x3A3A744E), AESx(0x0A0A141E),
	AESx(0x494992DB), AESx(0x06060C0A), AESx(0x2424486C), AESx(0x5C5CB8E4),
	AESx(0xC2C29F5D), AESx(0xD3D3BD6E), AESx(0xACAC43EF), AESx(0x6262C4A6),
	AESx(0x919139A8), AESx(0x959531A4), AESx(0xE4E4D337), AESx(0x7979F28B),
	AESx(0xE7E7D532), AESx(0xC8C88B43), AESx(0x37376E59), AESx(0x6D6DDAB7),
	AESx(0x8D8D018C), AESx(0xD5D5B164), AESx(0x4E4E9CD2), AESx(0xA9A949E0),
	AESx(0x6C6CD8B4), AESx(0x5656ACFA), AESx(0xF4F4F307), AESx(0xEAEACF25),
	AESx(0x6565CAAF), AESx(0x7A7AF48E), AESx(0xAEAE47E9), AESx(0x08081018),
	AESx(0xBABA6FD5), AESx(0x7878F088), AESx(0x25254A6F), AESx(0x2E2E5C72),
	AESx(0x1C1C3824), AESx(0xA6A657F1), AESx(0xB4B473C7), AESx(0xC6C69751),
	AESx(0xE8E8CB23), AESx(0xDDDDA17C), AESx(0x7474E89C), AESx(0x1F1F3E21),
	AESx(0x4B4B96DD), AESx(0xBDBD61DC), AESx(0x8B8B0D86), AESx(0x8A8A0F85),
	AESx(0x7070E090), AESx(0x3E3E7C42), AESx(0xB5B571C4), AESx(0x6666CCAA),
	AESx(0x484890D8), AESx(0x03030605), AESx(0xF6F6F701), AESx(0x0E0E1C12),
	AESx(0x6161C2A3), AESx(0x35356A5F), AESx(0x5757AEF9), AESx(0xB9B969D0),
	AESx(0x86861791), AESx(0xC1C19958), AESx(0x1D1D3A27), AESx(0x9E9E27B9),
	AESx(0xE1E1D938), AESx(0xF8F8EB13), AESx(0x98982BB3), AESx(0x11112233),
	AESx(0x6969D2BB), AESx(0xD9D9A970), AESx(0x8E8E0789), AESx(0x949433A7),
	AESx(0x9B9B2DB6), AESx(0x1E1E3C22), AESx(0x87871592), AESx(0xE9E9C920),
	AESx(0xCECE8749), AESx(0x5555AAFF), AESx(0x28285078), AESx(0xDFDFA57A),
	AESx(0x8C8C038F), AESx(0xA1A159F8), AESx(0x89890980), AESx(0x0D0D1A17),
	AESx(0xBFBF65DA), AESx(0xE6E6D731), AESx(0x424284C6), AESx(0x6868D0B8),
	AESx(0x414182C3), AESx(0x999929B0), AESx(0x2D2D5A77), AESx(0x0F0F1E11),
	AESx(0xB0B07BCB), AESx(0x5454A8FC), AESx(0xBBBB6DD6), AESx(0x16162C3A)
};

static const uint32_t h_AES2[256] = {
	AESx(0x63C6A563), AESx(0x7CF8847C), AESx(0x77EE9977), AESx(0x7BF68D7B),
	AESx(0xF2FF0DF2), AESx(0x6BD6BD6B), AESx(0x6FDEB16F), AESx(0xC59154C5),
	AESx(0x30605030), AESx(0x01020301), AESx(0x67CEA967), AESx(0x2B567D2B),
	AESx(0xFEE719FE), AESx(0xD7B562D7), AESx(0xAB4DE6AB), AESx(0x76EC9A76),
	AESx(0xCA8F45CA), AESx(0x821F9D82), AESx(0xC98940C9), AESx(0x7DFA877D),
	AESx(0xFAEF15FA), AESx(0x59B2EB59), AESx(0x478EC947), AESx(0xF0FB0BF0),
	AESx(0xAD41ECAD), AESx(0xD4B367D4), AESx(0xA25FFDA2), AESx(0xAF45EAAF),
	AESx(0x9C23BF9C), AESx(0xA453F7A4), AESx(0x72E49672), AESx(0xC09B5BC0),
	AESx(0xB775C2B7), AESx(0xFDE11CFD), AESx(0x933DAE93), AESx(0x264C6A26),
	AESx(0x366C5A36), AESx(0x3F7E413F), AESx(0xF7F502F7), AESx(0xCC834FCC),
	AESx(0x34685C34), AESx(0xA551F4A5), AESx(0xE5D134E5), AESx(0xF1F908F1),
	AESx(0x71E29371), AESx(0xD8AB73D8), AESx(0x31625331), AESx(0x152A3F15),
	AESx(0x04080C04), AESx(0xC79552C7), AESx(0x23466523), AESx(0xC39D5EC3),
	AESx(0x18302818), AESx(0x9637A196), AESx(0x050A0F05), AESx(0x9A2FB59A),
	AESx(0x070E0907), AESx(0x12243612), AESx(0x801B9B80), AESx(0xE2DF3DE2),
	AESx(0xEBCD26EB), AESx(0x274E6927), AESx(0xB27FCDB2), AESx(0x75EA9F75),
	AESx(0x09121B09), AESx(0x831D9E83), AESx(0x2C58742C), AESx(0x1A342E1A),
	AESx(0x1B362D1B), AESx(0x6EDCB26E), AESx(0x5AB4EE5A), AESx(0xA05BFBA0),
	AESx(0x52A4F652), AESx(0x3B764D3B), AESx(0xD6B761D6), AESx(0xB37DCEB3),
	AESx(0x29527B29), AESx(0xE3DD3EE3), AESx(0x2F5E712F), AESx(0x84139784),
	AESx(0x53A6F553), AESx(0xD1B968D1), AESx(0x00000000), AESx(0xEDC12CED),
	AESx(0x20406020), AESx(0xFCE31FFC), AESx(0xB179C8B1), AESx(0x5BB6ED5B),
	AESx(0x6AD4BE6A), AESx(0xCB8D46CB), AESx(0xBE67D9BE), AESx(0x39724B39),
	AESx(0x4A94DE4A), AESx(0x4C98D44C), AESx(0x58B0E858), AESx(0xCF854ACF),
	AESx(0xD0BB6BD0), AESx(0xEFC52AEF), AESx(0xAA4FE5AA), AESx(0xFBED16FB),
	AESx(0x4386C543), AESx(0x4D9AD74D), AESx(0x33665533), AESx(0x85119485),
	AESx(0x458ACF45), AESx(0xF9E910F9), AESx(0x02040602), AESx(0x7FFE817F),
	AESx(0x50A0F050), AESx(0x3C78443C), AESx(0x9F25BA9F), AESx(0xA84BE3A8),
	AESx(0x51A2F351), AESx(0xA35DFEA3), AESx(0x4080C040), AESx(0x8F058A8F),
	AESx(0x923FAD92), AESx(0x9D21BC9D), AESx(0x38704838), AESx(0xF5F104F5),
	AESx(0xBC63DFBC), AESx(0xB677C1B6), AESx(0xDAAF75DA), AESx(0x21426321),
	AESx(0x10203010), AESx(0xFFE51AFF), AESx(0xF3FD0EF3), AESx(0xD2BF6DD2),
	AESx(0xCD814CCD), AESx(0x0C18140C), AESx(0x13263513), AESx(0xECC32FEC),
	AESx(0x5FBEE15F), AESx(0x9735A297), AESx(0x4488CC44), AESx(0x172E3917),
	AESx(0xC49357C4), AESx(0xA755F2A7), AESx(0x7EFC827E), AESx(0x3D7A473D),
	AESx(0x64C8AC64), AESx(0x5DBAE75D), AESx(0x19322B19), AESx(0x73E69573),
	AESx(0x60C0A060), AESx(0x81199881), AESx(0x4F9ED14F), AESx(0xDCA37FDC),
	AESx(0x22446622), AESx(0x2A547E2A), AESx(0x903BAB90), AESx(0x880B8388),
	AESx(0x468CCA46), AESx(0xEEC729EE), AESx(0xB86BD3B8), AESx(0x14283C14),
	AESx(0xDEA779DE), AESx(0x5EBCE25E), AESx(0x0B161D0B), AESx(0xDBAD76DB),
	AESx(0xE0DB3BE0), AESx(0x32645632), AESx(0x3A744E3A), AESx(0x0A141E0A),
	AESx(0x4992DB49), AESx(0x060C0A06), AESx(0x24486C24), AESx(0x5CB8E45C),
	AESx(0xC29F5DC2), AESx(0xD3BD6ED3), AESx(0xAC43EFAC), AESx(0x62C4A662),
	AESx(0x9139A891), AESx(0x9531A495), AESx(0xE4D337E4), AESx(0x79F28B79),
	AESx(0xE7D532E7), AESx(0xC88B43C8), AESx(0x376E5937), AESx(0x6DDAB76D),
	AESx(0x8D018C8D), AESx(0xD5B164D5), AESx(0x4E9CD24E), AESx(0xA949E0A9),
	AESx(0x6CD8B46C), AESx(0x56ACFA56), AESx(0xF4F307F4), AESx(0xEACF25EA),
	AESx(0x65CAAF65), AESx(0x7AF48E7A), AESx(0xAE47E9AE), AESx(0x08101808),
	AESx(0xBA6FD5BA), AESx(0x78F08878), AESx(0x254A6F25), AESx(0x2E5C722E),
	AESx(0x1C38241C), AESx(0xA657F1A6), AESx(0xB473C7B4), AESx(0xC69751C6),
	AESx(0xE8CB23E8), AESx(0xDDA17CDD), AESx(0x74E89C74), AESx(0x1F3E211F),
	AESx(0x4B96DD4B), AESx(0xBD61DCBD), AESx(0x8B0D868B), AESx(0x8A0F858A),
	AESx(0x70E09070), AESx(0x3E7C423E), AESx(0xB571C4B5), AESx(0x66CCAA66),
	AESx(0x4890D848), AESx(0x03060503), AESx(0xF6F701F6), AESx(0x0E1C120E),
	AESx(0x61C2A361), AESx(0x356A5F35), AESx(0x57AEF957), AESx(0xB969D0B9),
	AESx(0x86179186), AESx(0xC19958C1), AESx(0x1D3A271D), AESx(0x9E27B99E),
	AESx(0xE1D938E1), AESx(0xF8EB13F8), AESx(0x982BB398), AESx(0x11223311),
	AESx(0x69D2BB69), AESx(0xD9A970D9), AESx(0x8E07898E), AESx(0x9433A794),
	AESx(0x9B2DB69B), AESx(0x1E3C221E), AESx(0x87159287), AESx(0xE9C920E9),
	AESx(0xCE8749CE), AESx(0x55AAFF55), AESx(0x28507828), AESx(0xDFA57ADF),
	AESx(0x8C038F8C), AESx(0xA159F8A1), AESx(0x89098089), AESx(0x0D1A170D),
	AESx(0xBF65DABF), AESx(0xE6D731E6), AESx(0x4284C642), AESx(0x68D0B868),
	AESx(0x4182C341), AESx(0x9929B099), AESx(0x2D5A772D), AESx(0x0F1E110F),
	AESx(0xB07BCBB0), AESx(0x54A8FC54), AESx(0xBB6DD6BB), AESx(0x162C3A16)
};

static const uint32_t h_AES3[256] = {
	AESx(0xC6A56363), AESx(0xF8847C7C), AESx(0xEE997777), AESx(0xF68D7B7B),
	AESx(0xFF0DF2F2), AESx(0xD6BD6B6B), AESx(0xDEB16F6F), AESx(0x9154C5C5),
	AESx(0x60503030), AESx(0x02030101), AESx(0xCEA96767), AESx(0x567D2B2B),
	AESx(0xE719FEFE), AESx(0xB562D7D7), AESx(0x4DE6ABAB), AESx(0xEC9A7676),
	AESx(0x8F45CACA), AESx(0x1F9D8282), AESx(0x8940C9C9), AESx(0xFA877D7D),
	AESx(0xEF15FAFA), AESx(0xB2EB5959), AESx(0x8EC94747), AESx(0xFB0BF0F0),
	AESx(0x41ECADAD), AESx(0xB367D4D4), AESx(0x5FFDA2A2), AESx(0x45EAAFAF),
	AESx(0x23BF9C9C), AESx(0x53F7A4A4), AESx(0xE4967272), AESx(0x9B5BC0C0),
	AESx(0x75C2B7B7), AESx(0xE11CFDFD), AESx(0x3DAE9393), AESx(0x4C6A2626),
	AESx(0x6C5A3636), AESx(0x7E413F3F), AESx(0xF502F7F7), AESx(0x834FCCCC),
	AESx(0x685C3434), AESx(0x51F4A5A5), AESx(0xD134E5E5), AESx(0xF908F1F1),
	AESx(0xE2937171), AESx(0xAB73D8D8), AESx(0x62533131), AESx(0x2A3F1515),
	AESx(0x080C0404), AESx(0x9552C7C7), AESx(0x46652323), AESx(0x9D5EC3C3),
	AESx(0x30281818), AESx(0x37A19696), AESx(0x0A0F0505), AESx(0x2FB59A9A),
	AESx(0x0E090707), AESx(0x24361212), AESx(0x1B9B8080), AESx(0xDF3DE2E2),
	AESx(0xCD26EBEB), AESx(0x4E692727), AESx(0x7FCDB2B2), AESx(0xEA9F7575),
	AESx(0x121B0909), AESx(0x1D9E8383), AESx(0x58742C2C), AESx(0x342E1A1A),
	AESx(0x362D1B1B), AESx(0xDCB26E6E), AESx(0xB4EE5A5A), AESx(0x5BFBA0A0),
	AESx(0xA4F65252), AESx(0x764D3B3B), AESx(0xB761D6D6), AESx(0x7DCEB3B3),
	AESx(0x527B2929), AESx(0xDD3EE3E3), AESx(0x5E712F2F), AESx(0x13978484),
	AESx(0xA6F55353), AESx(0xB968D1D1), AESx(0x00000000), AESx(0xC12CEDED),
	AESx(0x40602020), AESx(0xE31FFCFC), AESx(0x79C8B1B1), AESx(0xB6ED5B5B),
	AESx(0xD4BE6A6A), AESx(0x8D46CBCB), AESx(0x67D9BEBE), AESx(0x724B3939),
	AESx(0x94DE4A4A), AESx(0x98D44C4C), AESx(0xB0E85858), AESx(0x854ACFCF),
	AESx(0xBB6BD0D0), AESx(0xC52AEFEF), AESx(0x4FE5AAAA), AESx(0xED16FBFB),
	AESx(0x86C54343), AESx(0x9AD74D4D), AESx(0x66553333), AESx(0x11948585),
	AESx(0x8ACF4545), AESx(0xE910F9F9), AESx(0x04060202), AESx(0xFE817F7F),
	AESx(0xA0F05050), AESx(0x78443C3C), AESx(0x25BA9F9F), AESx(0x4BE3A8A8),
	AESx(0xA2F35151), AESx(0x5DFEA3A3), AESx(0x80C04040), AESx(0x058A8F8F),
	AESx(0x3FAD9292), AESx(0x21BC9D9D), AESx(0x70483838), AESx(0xF104F5F5),
	AESx(0x63DFBCBC), AESx(0x77C1B6B6), AESx(0xAF75DADA), AESx(0x42632121),
	AESx(0x20301010), AESx(0xE51AFFFF), AESx(0xFD0EF3F3), AESx(0xBF6DD2D2),
	AESx(0x814CCDCD), AESx(0x18140C0C), AESx(0x26351313), AESx(0xC32FECEC),
	AESx(0xBEE15F5F), AESx(0x35A29797), AESx(0x88CC4444), AESx(0x2E391717),
	AESx(0x9357C4C4), AESx(0x55F2A7A7), AESx(0xFC827E7E), AESx(0x7A473D3D),
	AESx(0xC8AC6464), AESx(0xBAE75D5D), AESx(0x322B1919), AESx(0xE6957373),
	AESx(0xC0A06060), AESx(0x19988181), AESx(0x9ED14F4F), AESx(0xA37FDCDC),
	AESx(0x44662222), AESx(0x547E2A2A), AESx(0x3BAB9090), AESx(0x0B838888),
	AESx(0x8CCA4646), AESx(0xC729EEEE), AESx(0x6BD3B8B8), AESx(0x283C1414),
	AESx(0xA779DEDE), AESx(0xBCE25E5E), AESx(0x161D0B0B), AESx(0xAD76DBDB),
	AESx(0xDB3BE0E0), AESx(0x64563232), AESx(0x744E3A3A), AESx(0x141E0A0A),
	AESx(0x92DB4949), AESx(0x0C0A0606), AESx(0x486C2424), AESx(0xB8E45C5C),
	AESx(0x9F5DC2C2), AESx(0xBD6ED3D3), AESx(0x43EFACAC), AESx(0xC4A66262),
	AESx(0x39A89191), AESx(0x31A49595), AESx(0xD337E4E4), AESx(0xF28B7979),
	AESx(0xD532E7E7), AESx(0x8B43C8C8), AESx(0x6E593737), AESx(0xDAB76D6D),
	AESx(0x018C8D8D), AESx(0xB164D5D5), AESx(0x9CD24E4E), AESx(0x49E0A9A9),
	AESx(0xD8B46C6C), AESx(0xACFA5656), AESx(0xF307F4F4), AESx(0xCF25EAEA),
	AESx(0xCAAF6565), AESx(0xF48E7A7A), AESx(0x47E9AEAE), AESx(0x10180808),
	AESx(0x6FD5BABA), AESx(0xF0887878), AESx(0x4A6F2525), AESx(0x5C722E2E),
	AESx(0x38241C1C), AESx(0x57F1A6A6), AESx(0x73C7B4B4), AESx(0x9751C6C6),
	AESx(0xCB23E8E8), AESx(0xA17CDDDD), AESx(0xE89C7474), AESx(0x3E211F1F),
	AESx(0x96DD4B4B), AESx(0x61DCBDBD), AESx(0x0D868B8B), AESx(0x0F858A8A),
	AESx(0xE0907070), AESx(0x7C423E3E), AESx(0x71C4B5B5), AESx(0xCCAA6666),
	AESx(0x90D84848), AESx(0x06050303), AESx(0xF701F6F6), AESx(0x1C120E0E),
	AESx(0xC2A36161), AESx(0x6A5F3535), AESx(0xAEF95757), AESx(0x69D0B9B9),
	AESx(0x17918686), AESx(0x9958C1C1), AESx(0x3A271D1D), AESx(0x27B99E9E),
	AESx(0xD938E1E1), AESx(0xEB13F8F8), AESx(0x2BB39898), AESx(0x22331111),
	AESx(0xD2BB6969), AESx(0xA970D9D9), AESx(0x07898E8E), AESx(0x33A79494),
	AESx(0x2DB69B9B), AESx(0x3C221E1E), AESx(0x15928787), AESx(0xC920E9E9),
	AESx(0x8749CECE), AESx(0xAAFF5555), AESx(0x50782828), AESx(0xA57ADFDF),
	AESx(0x038F8C8C), AESx(0x59F8A1A1), AESx(0x09808989), AESx(0x1A170D0D),
	AESx(0x65DABFBF), AESx(0xD731E6E6), AESx(0x84C64242), AESx(0xD0B86868),
	AESx(0x82C34141), AESx(0x29B09999), AESx(0x5A772D2D), AESx(0x1E110F0F),
	AESx(0x7BCBB0B0), AESx(0xA8FC5454), AESx(0x6DD6BBBB), AESx(0x2C3A1616)
};

static __constant__ uint32_t d_AES0[256];
static __constant__ uint32_t d_AES1[256];
static __constant__ uint32_t d_AES2[256];
static __constant__ uint32_t d_AES3[256];

static void aes_cpu_init()
{
	cudaMemcpyToSymbol( d_AES0,
                        h_AES0,
                        sizeof(h_AES0),
                        0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol( d_AES1,
                        h_AES1,
                        sizeof(h_AES1),
                        0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol( d_AES2,
                        h_AES2,
                        sizeof(h_AES2),
                        0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol( d_AES3,
                        h_AES3,
                        sizeof(h_AES3),
                        0, cudaMemcpyHostToDevice);
}

static __device__ __forceinline__ void aes_gpu_init(uint32_t *sharedMemory)
{
	if(threadIdx.x < 256)
	{
		sharedMemory[threadIdx.x] = d_AES0[threadIdx.x];
		sharedMemory[threadIdx.x+256] = d_AES1[threadIdx.x];
		sharedMemory[threadIdx.x+512] = d_AES2[threadIdx.x];
		sharedMemory[threadIdx.x+768] = d_AES3[threadIdx.x];
	}
}

static __device__ __forceinline__ void aes_gpu_init_flex(uint32_t *sharedMemory)
{
	if (blockDim.x >= 256)
	{
		aes_gpu_init(sharedMemory);
	}
	else
	{
		int chunksize = 1 << (31 - __clz(blockDim.x));
		if (threadIdx.x < chunksize)
		{
			int chunks = 256/chunksize;

			for(int i=0;i<chunks;i++)
			{
				int index = (threadIdx.x * chunks) + i;
				sharedMemory[index + 0] = d_AES0[index];
				sharedMemory[index+256] = d_AES1[index];
				sharedMemory[index+512] = d_AES2[index];
				sharedMemory[index+768] = d_AES3[index];
			}
		}
	}
}

static __device__ __forceinline__ void aes_round_quad(
	const uint32_t *sharedMemory,
	uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
	uint32_t k, uint32_t &y)
{
	uint32_t idx0, idx1, idx2, idx3;
	idx0 = __byte_perm(x0, 0, 0x4440);
	idx1 = __byte_perm(x1, 0, 0x4441) + 256;
	idx2 = __byte_perm(x2, 0, 0x4442) + 512;
	idx3 = __byte_perm(x3, 0, 0x4443) + 768;
	y =sharedMemory[idx0] ^ 
		sharedMemory[idx1] ^ 
		sharedMemory[idx2] ^ 
		sharedMemory[idx3] ^ k;
}

static __device__ __forceinline__ void aes_round(
	const uint32_t *sharedMemory,
	uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, 
	uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3, 
	uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3)
{
	uint32_t idx0, idx1, idx2, idx3;

	idx0 = __byte_perm(x0, 0, 0x4440);
	idx1 = __byte_perm(x1, 0, 0x4441) + 256;
	idx2 = __byte_perm(x2, 0, 0x4442) + 512;
	idx3 = __byte_perm(x3, 0, 0x4443) + 768;
	y0 =sharedMemory[idx0] ^ 
		sharedMemory[idx1] ^ 
		sharedMemory[idx2] ^ 
		sharedMemory[idx3] ^ k0;

	idx0 = __byte_perm(x1, 0, 0x4440);
	idx1 = __byte_perm(x2, 0, 0x4441) + 256;
	idx2 = __byte_perm(x3, 0, 0x4442) + 512;
	idx3 = __byte_perm(x0, 0, 0x4443) + 768;

	y1 =sharedMemory[idx0] ^ 
		sharedMemory[idx1] ^ 
		sharedMemory[idx2] ^ 
		sharedMemory[idx3] ^ k1;

	idx0 = __byte_perm(x2, 0, 0x4440);
	idx1 = __byte_perm(x3, 0, 0x4441) + 256;
	idx2 = __byte_perm(x0, 0, 0x4442) + 512;
	idx3 = __byte_perm(x1, 0, 0x4443) + 768;
	y2 =sharedMemory[idx0] ^ 
		sharedMemory[idx1] ^ 
		sharedMemory[idx2] ^ 
		sharedMemory[idx3] ^ k2;

	idx0 = __byte_perm(x3, 0, 0x4440);
	idx1 = __byte_perm(x0, 0, 0x4441) + 256;
	idx2 = __byte_perm(x1, 0, 0x4442) + 512;
	idx3 = __byte_perm(x2, 0, 0x4443) + 768;
	y3 =sharedMemory[idx0] ^ 
		sharedMemory[idx1] ^ 
		sharedMemory[idx2] ^ 
		sharedMemory[idx3] ^ k3;
}

static __device__ __forceinline__ void aes_round(
	const uint32_t *sharedMemory,
	uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, 
	uint32_t k0, 
	uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3)
{
	uint32_t idx0, idx1, idx2, idx3;

	idx0 = __byte_perm(x0, 0, 0x4440);
	idx1 = __byte_perm(x1, 0, 0x4441) + 256;
	idx2 = __byte_perm(x2, 0, 0x4442) + 512;
	idx3 = __byte_perm(x3, 0, 0x4443) + 768;
	y0 =sharedMemory[idx0] ^ 
		sharedMemory[idx1] ^ 
		sharedMemory[idx2] ^ 
		sharedMemory[idx3] ^ 
		k0;

	idx0 = __byte_perm(x1, 0, 0x4440);
	idx1 = __byte_perm(x2, 0, 0x4441) + 256;
	idx2 = __byte_perm(x3, 0, 0x4442) + 512;
	idx3 = __byte_perm(x0, 0, 0x4443) + 768;

	y1 =sharedMemory[idx0] ^ 
		sharedMemory[idx1] ^ 
		sharedMemory[idx2] ^ 
		sharedMemory[idx3]; // ^k3

	idx0 = __byte_perm(x2, 0, 0x4440);
	idx1 = __byte_perm(x3, 0, 0x4441) + 256;
	idx2 = __byte_perm(x0, 0, 0x4442) + 512;
	idx3 = __byte_perm(x1, 0, 0x4443) + 768;
	y2 =sharedMemory[idx0] ^ 
		sharedMemory[idx1] ^ 
		sharedMemory[idx2] ^ 
		sharedMemory[idx3]; // ^k2

	idx0 = __byte_perm(x3, 0, 0x4440);
	idx1 = __byte_perm(x0, 0, 0x4441) + 256;
	idx2 = __byte_perm(x1, 0, 0x4442) + 512;
	idx3 = __byte_perm(x2, 0, 0x4443) + 768;
	y3 =sharedMemory[idx0] ^ 
		sharedMemory[idx1] ^ 
		sharedMemory[idx2] ^ 
		sharedMemory[idx3]; // ^k3
}

static __device__ __forceinline__ void aes_round(
	const uint32_t *sharedMemory,
	uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, 
	uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3)
{
	uint32_t idx0, idx1, idx2, idx3;

	idx0 = __byte_perm(x0, 0, 0x4440);
	idx1 = __byte_perm(x1, 0, 0x4441) + 256;
	idx2 = __byte_perm(x2, 0, 0x4442) + 512;
	idx3 = __byte_perm(x3, 0, 0x4443) + 768;
	y0 =sharedMemory[idx0] ^ 
		sharedMemory[idx1] ^ 
		sharedMemory[idx2] ^ 
		sharedMemory[idx3];

	idx0 = __byte_perm(x1, 0, 0x4440);
	idx1 = __byte_perm(x2, 0, 0x4441) + 256;
	idx2 = __byte_perm(x3, 0, 0x4442) + 512;
	idx3 = __byte_perm(x0, 0, 0x4443) + 768;

	y1 =sharedMemory[idx0] ^ 
		sharedMemory[idx1] ^ 
		sharedMemory[idx2] ^ 
		sharedMemory[idx3]; // ^k3

	idx0 = __byte_perm(x2, 0, 0x4440);
	idx1 = __byte_perm(x3, 0, 0x4441) + 256;
	idx2 = __byte_perm(x0, 0, 0x4442) + 512;
	idx3 = __byte_perm(x1, 0, 0x4443) + 768;
	y2 =sharedMemory[idx0] ^ 
		sharedMemory[idx1] ^ 
		sharedMemory[idx2] ^ 
		sharedMemory[idx3]; // ^k2

	idx0 = __byte_perm(x3, 0, 0x4440);
	idx1 = __byte_perm(x0, 0, 0x4441) + 256;
	idx2 = __byte_perm(x1, 0, 0x4442) + 512;
	idx3 = __byte_perm(x2, 0, 0x4443) + 768;
	y3 =sharedMemory[idx0] ^ 
		sharedMemory[idx1] ^ 
		sharedMemory[idx2] ^ 
		sharedMemory[idx3]; // ^k3
}


///////////// AES ohne shared memory
static const uint32_t FSb[64] =
{
    0x7B777C63, 0xC56F6BF2, 0x2B670130, 0x76ABD7FE, 
	0x7DC982CA, 0xF04759FA, 0xAFA2D4AD, 0xC072A49C, 
	0x2693FDB7, 0xCCF73F36, 0xF1E5A534, 0x1531D871, 
	0xC323C704, 0x9A059618, 0xE2801207, 0x75B227EB, 
	0x1A2C8309, 0xA05A6E1B, 0xB3D63B52, 0x842FE329, 
	0xED00D153, 0x5BB1FC20, 0x39BECB6A, 0xCF584C4A, 
	0xFBAAEFD0, 0x85334D43, 0x7F02F945, 0xA89F3C50, 
	0x8F40A351, 0xF5389D92, 0x21DAB6BC, 0xD2F3FF10, 
	0xEC130CCD, 0x1744975F, 0x3D7EA7C4, 0x73195D64, 
	0xDC4F8160, 0x88902A22, 0x14B8EE46, 0xDB0B5EDE, 
	0x0A3A32E0, 0x5C240649, 0x62ACD3C2, 0x79E49591, 
	0x6D37C8E7, 0xA94ED58D, 0xEAF4566C, 0x08AE7A65, 
	0x2E2578BA, 0xC6B4A61C, 0x1F74DDE8, 0x8A8BBD4B, 
	0x66B53E70, 0x0EF60348, 0xB9573561, 0x9E1DC186, 
	0x1198F8E1, 0x948ED969, 0xE9871E9B, 0xDF2855CE, 
	0x0D89A18C, 0x6842E6BF, 0x0F2D9941, 0x16BB54B0
};

static __constant__ uint32_t d_AES_FSb[64];

static void aes_reg_cpu_init()
{
	cudaMemcpyToSymbol( d_AES_FSb,
                        FSb,
                        sizeof(FSb),
                        0, cudaMemcpyHostToDevice);
}

static __device__ __forceinline__ void aes_reg_gpu_init(uint32_t *regs)
{
	int idx = (threadIdx.x & 0x1F)<<1;
	regs[0] = d_AES_FSb[idx+0];
	regs[1] = d_AES_FSb[idx+1];
}

// Diese funktionen dienen nur als beispiel und müssen selbst implementiert werden!!!
static __device__ __forceinline__ uint32_t aes_reg_getByIndex(int idx, uint32_t r0, uint32_t r1)
{
	int regThreadIndex = idx>>3;
	int regByteIndex = idx & 0x07;

	uint32_t t0 = __shfl((int)r0, regThreadIndex, 32);
	uint32_t t1 = __shfl((int)r1, regThreadIndex, 32);

	uint32_t x = __byte_perm(t0, t1, regByteIndex) & 0xFF;
	uint32_t y = ( ( x << 1 ) ^ ( ( x & 0x80 ) ? 0x1B : 0x00 ) ) & 0xFF;
    uint32_t z =  ( y ^ x );

	return y | __byte_perm(x, z, 0x4001);

	/*
	return	( y       ) ^
            ( x <<  8 ) ^
            ( x << 16 ) ^
            ( z << 24 );
	*/
}

static __device__ __forceinline__ void aes_reg_round_quad(
	uint32_t r0, uint32_t r1,
	uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
	uint32_t k, uint32_t &y)
{
	uint32_t idx0, idx1, idx2, idx3;
	idx0 = __byte_perm(x0, 0, 0x4440);
	idx1 = __byte_perm(x1, 0, 0x4441);
	idx2 = __byte_perm(x2, 0, 0x4442);
	idx3 = __byte_perm(x3, 0, 0x4443);

	idx0 = aes_reg_getByIndex(idx0, r0, r1);
	idx1 = aes_reg_getByIndex(idx1, r0, r1);
	idx2 = aes_reg_getByIndex(idx2, r0, r1);
	idx3 = aes_reg_getByIndex(idx3, r0, r1);

	y = idx0 ^
		__byte_perm(idx1, 0, 0x2103) ^
		__byte_perm(idx2, 0, 0x1032) ^
		__byte_perm(idx3, 0, 0x0321) ^ k;
}

static __device__ __forceinline__ void aes_reg_round(
	uint32_t r0, uint32_t r1,
	uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, 
	uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3, 
	uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3)
{
	uint32_t idx0, idx1, idx2, idx3;

	// 0
	idx0 = __byte_perm(x0, 0, 0x4440);
	idx1 = __byte_perm(x1, 0, 0x4441);
	idx2 = __byte_perm(x2, 0, 0x4442);
	idx3 = __byte_perm(x3, 0, 0x4443);

	idx0 = aes_reg_getByIndex(idx0, r0, r1);
	idx1 = aes_reg_getByIndex(idx1, r0, r1);
	idx2 = aes_reg_getByIndex(idx2, r0, r1);
	idx3 = aes_reg_getByIndex(idx3, r0, r1);

	y0 = idx0 ^
		__byte_perm(idx1, 0, 0x2103) ^
		__byte_perm(idx2, 0, 0x1032) ^
		__byte_perm(idx3, 0, 0x0321) ^ k0;

	// 1
	idx0 = __byte_perm(x1, 0, 0x4440);
	idx1 = __byte_perm(x2, 0, 0x4441);
	idx2 = __byte_perm(x3, 0, 0x4442);
	idx3 = __byte_perm(x0, 0, 0x4443);

	idx0 = aes_reg_getByIndex(idx0, r0, r1);
	idx1 = aes_reg_getByIndex(idx1, r0, r1);
	idx2 = aes_reg_getByIndex(idx2, r0, r1);
	idx3 = aes_reg_getByIndex(idx3, r0, r1);

	y1 = idx0 ^
		__byte_perm(idx1, 0, 0x2103) ^
		__byte_perm(idx2, 0, 0x1032) ^
		__byte_perm(idx3, 0, 0x0321) ^ k1;

	// 2
	idx0 = __byte_perm(x2, 0, 0x4440);
	idx1 = __byte_perm(x3, 0, 0x4441);
	idx2 = __byte_perm(x0, 0, 0x4442);
	idx3 = __byte_perm(x1, 0, 0x4443);

	idx0 = aes_reg_getByIndex(idx0, r0, r1);
	idx1 = aes_reg_getByIndex(idx1, r0, r1);
	idx2 = aes_reg_getByIndex(idx2, r0, r1);
	idx3 = aes_reg_getByIndex(idx3, r0, r1);

	y2 = idx0 ^
		__byte_perm(idx1, 0, 0x2103) ^
		__byte_perm(idx2, 0, 0x1032) ^
		__byte_perm(idx3, 0, 0x0321) ^ k2;


	// 3
	idx0 = __byte_perm(x3, 0, 0x4440);
	idx1 = __byte_perm(x0, 0, 0x4441);
	idx2 = __byte_perm(x1, 0, 0x4442);
	idx3 = __byte_perm(x2, 0, 0x4443);

	idx0 = aes_reg_getByIndex(idx0, r0, r1);
	idx1 = aes_reg_getByIndex(idx1, r0, r1);
	idx2 = aes_reg_getByIndex(idx2, r0, r1);
	idx3 = aes_reg_getByIndex(idx3, r0, r1);

	y3 = idx0 ^
		__byte_perm(idx1, 0, 0x2103) ^
		__byte_perm(idx2, 0, 0x1032) ^
		__byte_perm(idx3, 0, 0x0321) ^ k3;
}

// global memory uint32_t read for streaming operation
__device__ __forceinline__ uint32_t load_global_cs(const uint32_t *addr)
{
    uint32_t value;
    asm("ld.global.cs.u32 %0, [%1];" : "=r"(value) : "l"(addr) );
    return value;
}

// global memory uint32_t write for streaming operation
__device__ __forceinline__ void store_global_cs(const uint32_t *addr, uint32_t val)
{
    asm("st.global.cs.u32 [%0], %1;" :: "l"(addr) "r"(val) );
}

__device__ __forceinline__ int CalculateMemoryPosition(uint32_t *addr)
{
	// calculate the position of the element
	int pos = (int)((addr[0] >> 4) & 0x1ffff);
	// return exact element (in uint32_t, not in bytes)
	// j * AES_BLOCK_SIZE = j * 16 (in bytes) = j * 4 (in uint32_t's)
	
	return pos<<2;
}

// TODO: check the code posted here if it is more efficient
// http://stackoverflow.com/questions/6162140/128-bit-integer-on-cuda

#if AES_WITHOUT_SHARED
__device__ __forceinline__ void EncryptAES_ECB(uint32_t r0, uint32_t r1, const uint32_t *roundKeys, uint32_t *state)
{
#pragma unroll 5
	for(int i=0;i<5;i++)
	{
		int rkOffset = i<<3;
		uint32_t temp[4];

		aes_reg_round(r0, r1,
			state[0], state[1], state[2], state[3],
			roundKeys[0+rkOffset], roundKeys[1+rkOffset], roundKeys[2+rkOffset], roundKeys[3+rkOffset], 
			temp[0], temp[1], temp[2], temp[3]);

		aes_reg_round(r0, r1,
			temp[0], temp[1], temp[2], temp[3],
			roundKeys[4+rkOffset], roundKeys[5+rkOffset], roundKeys[6+rkOffset], roundKeys[7+rkOffset],
			state[0], state[1], state[2], state[3]);
	}
}
#else
__device__ __forceinline__ void EncryptAES_ECB(const uint32_t *sharedMemory, const uint32_t *roundKeys, uint32_t *state)
{
#pragma unroll 5
	for(int i=0;i<5;i++)
	{
		int rkOffset = i<<3;
		uint32_t temp[4];

		aes_round(sharedMemory,
			state[0], state[1], state[2], state[3],
			roundKeys[0+rkOffset], roundKeys[1+rkOffset], roundKeys[2+rkOffset], roundKeys[3+rkOffset], 
			temp[0], temp[1], temp[2], temp[3]);

		aes_round(sharedMemory,
			temp[0], temp[1], temp[2], temp[3],
			roundKeys[4+rkOffset], roundKeys[5+rkOffset], roundKeys[6+rkOffset], roundKeys[7+rkOffset],
			state[0], state[1], state[2], state[3]);
	}
}
#endif


__global__ void cuda_mro_hash_prepare_octo(int threads,
					   uint32_t *g_prepMemory, uint32_t *g_keccakOutputData,
					   uint32_t *g_keccakOutputExpandedAESKey,
					   uint32_t *g_abInput, uint32_t *g_ArrAB)
{
#if AES_WITHOUT_SHARED
	uint32_t aes[2];
	aes_reg_gpu_init(aes);
#else
	__shared__ uint32_t sharedMemory[1024];
	aes_gpu_init_flex(sharedMemory);
#endif
	int thread = (blockDim.x * blockIdx.x + threadIdx.x)>>3;
    if (thread < threads)
	{
		int octId = (threadIdx.x & 0x07);
		////// Keccak is precalculated		

		uint32_t text[4], key[40];
		uint32_t thr34 = thread*34;
		uint32_t thr40 = thread*40;		
		uint32_t thr16 = (thread<<4);
		uint32_t thr32 = (thread<<5);

#if DATA_LAYOUT_SWITCH
		uint32_t thread4 = (thread<<2);
#else
		uint32_t thr2MB = (thread<<19);		
#endif

		// copy text & key
		#pragma unroll 4
		for(int j=0;j<4;j++)
			text[j] = g_keccakOutputData[thr34 + j + (octId<<2)];
		#pragma unroll 40
		for(int j=0;j<40;j++)
			key[j] = g_keccakOutputExpandedAESKey[thr40 + j];

		for(int i=0;i<0x4000;i++) // 16k
		{
			// text will be updated
#if AES_WITHOUT_SHARED
			EncryptAES_ECB(aes[0], aes[1], key, text);
#else
			EncryptAES_ECB(sharedMemory, key, text);			
#endif
			// copy to output memory
			// compute index:
			// AES_BLOCKSIZE = 16, INIT_SIZE_BYTE = 128
			// (j * AES_BLOCKSIZE) + (i * INIT_SIZE_BYTE) ^= Bytes
			// (octId << 4) + (i << 7) ^= Bytes
			// (octId << 2) + (i << 5) ^= uint32_t's						
			
			//int offset = (octId << 2) + (i << 5);
			//#pragma unroll 4
			//for(int j=0;j<4;j++)
				//g_prepMemory[thr2MB + offset + j] = text[j];

			// new Data Layout
			// 16x uint32_t per thread (one AES_BLOCK)
			//  4x uint32_t per octID (one quad of AES_BLOCK)
			// stride = number of threads
			// full size = number of threads * 16
#if DATA_LAYOUT_SWITCH
			int offset = (octId<<2) + (i << 5);
			#pragma unroll 4
			for(int j=0;j<4;j++)
				g_prepMemory[j + (thread4) + (offset*threads) ] = text[j];
#else
			int offset = (octId << 2) + (i << 5);
			#pragma unroll 4
			for(int j=0;j<4;j++)
				g_prepMemory[thr2MB + offset + j] = text[j];
#endif
		}

		/////// Write a + b parallel
		uint32_t t1, t2;
		t1 = g_abInput[thr16 + octId];
		t2 = g_abInput[thr16 + octId + 8];

		uint32_t temp =	t1 ^ t2;
		g_ArrAB[thr32 + octId] = temp;
	}
}

__global__ void cuda_mro_hash_post_octo(int threads, uint32_t *g_prepMemory, 
							uint32_t *g_keccakOutputData, uint32_t *g_keccakOutputExpandedAESKey,
							uint32_t *g_hashOutput)
{
#if AES_WITHOUT_SHARED
	uint32_t aes[2];
	aes_reg_gpu_init(aes);
#else
	__shared__ uint32_t sharedMemory[1024];
	aes_gpu_init_flex(sharedMemory);
#endif

	int thread = (blockDim.x * blockIdx.x + threadIdx.x)>>3;
    if (thread < threads)
	{
		int octId = (threadIdx.x & 0x07);
		uint32_t text[4], key[40];
		uint32_t thr34 = thread*34;
		uint32_t thr40 = thread*40;
		uint32_t thr32 = (thread<<5);		

#if DATA_LAYOUT_SWITCH
		uint32_t thread4 = (thread<<2);
#else
		uint32_t thr2MB = (thread<<19);		
#endif

		// copy text & key
		#pragma unroll 4
		for(int j=0;j<4;j++)
			text[j] = g_keccakOutputData[thr34 + j + (octId<<2)];

		#pragma unroll 40
		for(int j=0;j<40;j++)
			key[j] = g_keccakOutputExpandedAESKey[thr40 + j];

		for(int i=0;i<0x4000;i++) // 16k
		{
			// compute index:
			// AES_BLOCKSIZE = 16, INIT_SIZE_BYTE = 128
			// (j * AES_BLOCKSIZE) + (i * INIT_SIZE_BYTE) ^= Bytes
			// (octId << 4) + (i << 7) ^= Bytes
			// (octId << 2) + (i << 5) ^= uint32_t's
			//int offset = (octId << 2) + (i << 5);
			uint32_t block[4];

			// Load Blockdata
			//#pragma unroll 4
			//for(int j=0;j<4;j++)
			//	block[j] = g_prepMemory[thr2MB + offset + j];
			// new Data Layout			
#if DATA_LAYOUT_SWITCH
			int offset = (octId<<2) + (i << 5);
			#pragma unroll 4
			for(int j=0;j<4;j++)
				block[j] = g_prepMemory[j + (thread4) + (offset*threads) ];
#else
			int offset = (octId << 2) + (i << 5);
			#pragma unroll 4
			for(int j=0;j<4;j++)
				block[j] = g_prepMemory[thr2MB + offset + j];
#endif

			// XOR
#pragma unroll 4
			for(int j=0;j<4;j++)
				text[j] ^= block[j];

			// do aes, text will be updated
#if AES_WITHOUT_SHARED
			EncryptAES_ECB(aes[0], aes[1], key, text);
#else			
			EncryptAES_ECB(sharedMemory, key, text);
#endif
		}

		// Output the Hash
		#pragma unroll 4
		for(int j=0;j<4;j++)
			g_hashOutput[thr32 + (octId << 2) + j] = text[j];
	}
}

__global__ void cuda_mro_hash_doWork_quad(int threads, uint32_t *g_prepMemory, uint32_t *g_ArrAB)
{
#if AES_WITHOUT_SHARED
	uint32_t aes[2];
	aes_reg_gpu_init(aes);
#else
	__shared__ uint32_t sharedMemory[1024];
	aes_gpu_init_flex(sharedMemory);
#endif

	int thread = (blockDim.x * blockIdx.x + threadIdx.x)>>2;
    if (thread < threads)
    {
		uint32_t a, b;
		uint32_t c[4];
		uint32_t d;
		uint32_t thr32 = (thread<<5);
#if DATA_LAYOUT_SWITCH
		uint32_t thread4 = (thread<<2);
#else
		uint32_t thr2MB = (thread<<19);		
#endif
		int quadId = (threadIdx.x & 0x03);
		// read a,b from global memory
		a = g_ArrAB[thr32 + quadId];
		b = g_ArrAB[thr32 + quadId + 4];

		for(int i=0;i<0x80000;i++)
		{
			// calculate position
			uint32_t posBase = __shfl((int)a, 0, 4);
			int position = CalculateMemoryPosition(&posBase);

			// new Data Layout
#if DATA_LAYOUT_SWITCH
//			c[0] = g_prepMemory[quadId + thread4 + (position*threads)];
			c[0] = load_global_cs(&g_prepMemory[quadId + thread4 + (position*threads)]);
#else
//			c[0] = g_prepMemory[thr2MB + position + quadId];
			c[0] = load_global_cs(&g_prepMemory[thr2MB + position + quadId]);
#endif

			// to let this function run correctly, earch quad-thread
			// needs an other order of c.
			// The first thread has  c0, c1, c2, c3
			// The second thread has c1, c2, c3, c0
			// The thrird thread has c2, c3, c0, c1
			// The fourth thread has c3, c0, c1, c2
			c[1] = __shfl((int)c[0], quadId + 1, 4);
			c[2] = __shfl((int)c[0], quadId + 2, 4);
			c[3] = __shfl((int)c[0], quadId + 3, 4);
#if AES_WITHOUT_SHARED
			aes_reg_round_quad(aes[0], aes[1],
				c[0], c[1], c[2], c[3],
				a,
				d);
#else
			aes_round_quad(sharedMemory,
				c[0], c[1], c[2], c[3],
				a,
				d);
#endif

			// XOR
			b ^= d;

			// store
#if DATA_LAYOUT_SWITCH
//			g_prepMemory[quadId + thread4 + (position*threads)] = b;
			store_global_cs(&g_prepMemory[quadId + thread4 + (position*threads)], b);
#else
//			g_prepMemory[thr2MB + position + quadId] = b;
			store_global_cs(&g_prepMemory[thr2MB + position + quadId], b);
#endif

			// search new Position
			posBase = __shfl((int)d, 0, 4);
			position = CalculateMemoryPosition(&posBase);

			// read from this position
#if DATA_LAYOUT_SWITCH
//			b = g_prepMemory[quadId + thread4 + (position*threads)];
			b = load_global_cs(&g_prepMemory[quadId + thread4 + (position*threads)]);
#else
//			b = g_prepMemory[thr2MB + position + quadId];
			b = load_global_cs(&g_prepMemory[thr2MB + position + quadId]);
#endif

			// multiply

			// we want to calculate b * d.			
			uint32_t m1[2], m2[2];
			// b
			m1[0] = __shfl((int)b, 0, 4);
			m1[1] = __shfl((int)b, 1, 4);

			// d
			m2[0] = __shfl((int)d, 0, 4);
			m2[1] = __shfl((int)d, 1, 4);			

			// now, thread 1 has b in m1 and thread 3 has d in m2
			uint64_t n1, n2;
			n1 = *(const uint64_t *)m1;
			n2 = *(const uint64_t *)m2;

			uint64_t pResLo = n1 * n2; // thread 2,3
			uint64_t pResHi = __umul64hi(n1, n2); // thread 0,1
			uint64_t pRes = (quadId & 0x02) ? pResLo : pResHi;

			// pRes to uint32_t
			m1[0] = LOWORD(pRes);
			m1[1] = HIWORD(pRes);

			// shuffle
			m2[0] = __shfl((int)m1[1], quadId-1, 4);

			// save
			c[0] = (quadId & 0x01) ? m2[0] : m1[0];

			uint32_t carry;
			c[0] += a;
			carry = (c[0] < a);
			carry = __shfl((int)carry, quadId - 1, 4);
			c[0] += (quadId & 0x01) ? carry : 0;			

			// store
#if DATA_LAYOUT_SWITCH
//			g_prepMemory[quadId + thread4 + (position*threads)] = c[0];
			store_global_cs(&g_prepMemory[quadId + thread4 + (position*threads)], c[0]);
#else
//			g_prepMemory[thr2MB + position + quadId] = c[0];
			store_global_cs(&g_prepMemory[thr2MB + position + quadId], c[0]);
#endif

			// XOR + move c to b
			a = b ^ c[0];
			b = d;
		}
	}
}

#include <stdio.h>

__host__ int cryptonight_num_smx(int thr_id)
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, thr_id);
    return props.multiProcessorCount;
}

// Setup-Funktionen
__host__ void cryptonight_cpu_init(int thr_id, int threads)
{
	cudaSetDevice(thr_id);

	aes_cpu_init();
	aes_reg_cpu_init();

	size_t mem = 2 * 1024 * 1024;
	mem = mem * (size_t)threads;
	d_prepMemory[thr_id] = NULL;
	cudaMalloc(&d_prepMemory[thr_id], mem);

	d_keccakOutputExpandedAESKey1[thr_id] = NULL;
	cudaMalloc(&d_keccakOutputExpandedAESKey1[thr_id], 160 * threads);

	d_keccakOutputExpandedAESKey2[thr_id] = NULL;
	cudaMalloc(&d_keccakOutputExpandedAESKey2[thr_id], 160 * threads);

	d_keccakOutputData[thr_id] = NULL;
	cudaMalloc(&d_keccakOutputData[thr_id], 136 * threads);

	d_hashOutput[thr_id] = NULL;
	cudaMalloc(&d_hashOutput[thr_id], 128 * threads);

	d_abInput[thr_id] = NULL;
	cudaMalloc(&d_abInput[thr_id], 64 * threads);

	if (d_prepMemory[thr_id] == NULL || d_keccakOutputExpandedAESKey1[thr_id] == NULL || d_keccakOutputExpandedAESKey2[thr_id] == NULL ||
		d_keccakOutputData[thr_id] == NULL || d_hashOutput[thr_id] == NULL || d_abInput[thr_id] == NULL) {
		fprintf(stderr, "Out of memory!\n");
		exit(1);
	}

}

__host__ void cryptonight_cpu_hash_test_flex(int thr_id, int threads, uint32_t *h_keccakOutputData, 
		uint32_t *h_keccakOutputExpandedAESKey1, uint32_t *h_keccakOutputExpandedAESKey2,
		uint32_t *h_abInput, uint32_t *h_hashOutput, int order, const int threadsperblock)
{
	cudaMemcpy(d_keccakOutputExpandedAESKey1[thr_id], h_keccakOutputExpandedAESKey1, 160 * threads, cudaMemcpyHostToDevice);
	cudaMemcpy(d_keccakOutputExpandedAESKey2[thr_id], h_keccakOutputExpandedAESKey2, 160 * threads, cudaMemcpyHostToDevice);
	cudaMemcpy(d_keccakOutputData[thr_id], h_keccakOutputData, 136 * threads, cudaMemcpyHostToDevice);
	cudaMemcpy(d_abInput[thr_id], h_abInput, 64 * threads, cudaMemcpyHostToDevice);

#if 0
	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
#endif

	// quad kernel quadruples the amount of threads per block
	dim3 gridQuad((threads + threadsperblock-1)/threadsperblock);
	dim3 blockQuad(4*threadsperblock);

	// the octo kernels use 4*threadsperblock and threads*2 for the grid
	// to achieve an expansion of factor 8
	dim3 gridOcto(((threads*2) + threadsperblock-1)/threadsperblock);
	dim3 blockOcto(4*threadsperblock);

	// Größe des dynamischen Shared Memory Bereichs
	size_t shared_size = 0;

	// hashOutput will be used as temporary memory here
	cuda_mro_hash_prepare_octo<<<gridOcto, blockOcto, shared_size>>>(threads, d_prepMemory[thr_id], d_keccakOutputData[thr_id], d_keccakOutputExpandedAESKey1[thr_id], d_abInput[thr_id], d_hashOutput[thr_id]);
	MyStreamSynchronize(NULL, order, thr_id);
//	cuda_mro_hash_doWork<<<grid, block, shared_size>>>(threads, d_prepMemory[thr_id], d_hashOutput[thr_id]);
	cuda_mro_hash_doWork_quad<<<gridQuad, blockQuad, shared_size>>>(threads, d_prepMemory[thr_id], d_hashOutput[thr_id]);
	MyStreamSynchronize(NULL, order+1, thr_id);
	cuda_mro_hash_post_octo<<<gridOcto, blockOcto, shared_size>>>(threads, d_prepMemory[thr_id], d_keccakOutputData[thr_id], d_keccakOutputExpandedAESKey2[thr_id], d_hashOutput[thr_id]);
	MyStreamSynchronize(NULL, order+2, thr_id);

	//cudaMemcpy(h_hashOutput, d_prepMemory[thr_id], 1<<21, cudaMemcpyDeviceToHost);	
	cudaMemcpy(h_hashOutput, d_hashOutput[thr_id], 128 * threads, cudaMemcpyDeviceToHost);	
}

#include <map>
#include <sys/time.h>
#include <unistd.h>

// Zeitsynchronisations-Routine von cudaminer mit CPU sleep
typedef struct { double value[8]; } tsumarray;
cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id)
{
    cudaError_t result = cudaSuccess;
    if (situation >= 0)
    {   
        static std::map<int, tsumarray> tsum;

        double a = 0.95, b = 0.05;
        if (tsum.find(situation) == tsum.end()) { a = 0.5; b = 0.5; } // faster initial convergence

        double tsync = 0.0;
        double tsleep = 0.95 * tsum[situation].value[thr_id];
        if (cudaStreamQuery(stream) == cudaErrorNotReady)
        {
            usleep((useconds_t)(1e6*tsleep));
            struct timeval tv_start, tv_end;
            gettimeofday(&tv_start, NULL);
            result = cudaStreamSynchronize(stream);
            gettimeofday(&tv_end, NULL);
            tsync = 1e-6 * (tv_end.tv_usec-tv_start.tv_usec) + (tv_end.tv_sec-tv_start.tv_sec);
        }
        if (tsync >= 0) tsum[situation].value[thr_id] = a * tsum[situation].value[thr_id] + b * (tsleep+tsync);
    }
    else
        result = cudaStreamSynchronize(stream);
    return result;
}

