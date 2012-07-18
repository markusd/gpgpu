#ifndef DC_UTIL_HPP_
#define DC_UTIL_HPP_

#include <D3D/d3dcommon.h>
#include <D3D/d3d11.h>
#include <D3D/d3dcompiler.h>
#include <D3D/d3dx11.h>

namespace dc {

HRESULT CreateComputeDevice(ID3D11Device** ppDeviceOut, ID3D11DeviceContext** ppContextOut, bool bForceRef);
HRESULT CreateComputeShader(LPCWSTR pSrcFile, LPCSTR pFunctionName, ID3D11Device* pDevice, ID3D11ComputeShader** ppShaderOut);

}

#endif