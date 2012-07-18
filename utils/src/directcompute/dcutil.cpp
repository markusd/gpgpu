#include <directcompute/dcutil.hpp>
#include <iostream>

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)      { if (p) { (p)->Release(); (p)=NULL; } }
#endif

namespace dc {

HRESULT WINAPI Dynamic_D3D11CreateDevice( IDXGIAdapter* pAdapter,
                                          D3D_DRIVER_TYPE DriverType,
                                          HMODULE Software,
                                          UINT32 Flags,
                                          CONST D3D_FEATURE_LEVEL* pFeatureLevels,
                                          UINT FeatureLevels,
                                          UINT32 SDKVersion,
                                          ID3D11Device** ppDevice,
                                          D3D_FEATURE_LEVEL* pFeatureLevel,
                                          ID3D11DeviceContext** ppImmediateContext)
{
    typedef HRESULT (WINAPI * LPD3D11CREATEDEVICE)( IDXGIAdapter*, D3D_DRIVER_TYPE, HMODULE, UINT32, CONST D3D_FEATURE_LEVEL*, UINT, UINT32, ID3D11Device**, D3D_FEATURE_LEVEL*, ID3D11DeviceContext** );
    static LPD3D11CREATEDEVICE  s_DynamicD3D11CreateDevice = NULL;
    
    if ( s_DynamicD3D11CreateDevice == NULL )
    {            
        HMODULE hModD3D11 = LoadLibrary( "d3d11.dll" );

        if ( hModD3D11 == NULL )
        {
            // Ensure this "D3D11 absent" message is shown only once. As sometimes, the app would like to try
            // to create device multiple times
            static bool bMessageAlreadyShwon = false;
            
            if ( !bMessageAlreadyShwon )
            {
                OSVERSIONINFOEX osv;
                memset( &osv, 0, sizeof(osv) );
                osv.dwOSVersionInfoSize = sizeof(osv);
                GetVersionEx( (LPOSVERSIONINFO)&osv );

                if ( ( osv.dwMajorVersion > 6 )
                    || ( osv.dwMajorVersion == 6 && osv.dwMinorVersion >= 1 ) 
                    || ( osv.dwMajorVersion == 6 && osv.dwMinorVersion == 0 && osv.dwBuildNumber > 6002 ) )
                {

					std::cout << "Error: Direct3D 11 components were not found." << std::endl;
                    // This should not happen, but is here for completeness as the system could be
                    // corrupted or some future OS version could pull D3D11.DLL for some reason
                }
                else if ( osv.dwMajorVersion == 6 && osv.dwMinorVersion == 0 && osv.dwBuildNumber == 6002 )
                {
                    std::cout << "Error: Direct3D 11 components were not found, but are available for"\
                        " this version of Windows.\n"\
                        "For details see Microsoft Knowledge Base Article #971644\n"\
                        "http://support.microsoft.com/default.aspx/kb/971644/" << std::endl;

                }
                else if ( osv.dwMajorVersion == 6 && osv.dwMinorVersion == 0 )
                {
                   std::cout << "Error: Direct3D 11 components were not found. Please install the latest Service Pack.\n"\
                        "For details see Microsoft Knowledge Base Article #935791\n"\
                        " http://support.microsoft.com/default.aspx/kb/935791" << std::endl;

                }
                else
                {
                   std::cout << "Error: Direct3D 11 is not supported on this OS." << std::endl;
                }

                bMessageAlreadyShwon = true;
            }            

            return E_FAIL;
        }

        s_DynamicD3D11CreateDevice = ( LPD3D11CREATEDEVICE )GetProcAddress( hModD3D11, "D3D11CreateDevice" );           
    }

    return s_DynamicD3D11CreateDevice( pAdapter, DriverType, Software, Flags, pFeatureLevels, FeatureLevels,
                                       SDKVersion, ppDevice, pFeatureLevel, ppImmediateContext );
}

//--------------------------------------------------------------------------------------
// Create the D3D device and device context suitable for running Compute Shaders(CS)
//--------------------------------------------------------------------------------------
HRESULT CreateComputeDevice( ID3D11Device** ppDeviceOut, ID3D11DeviceContext** ppContextOut, bool bForceRef )
{    
    *ppDeviceOut = NULL;
    *ppContextOut = NULL;
    
    HRESULT hr = S_OK;

    UINT uCreationFlags = D3D11_CREATE_DEVICE_SINGLETHREADED;
#if defined(DEBUG) || defined(_DEBUG)
    uCreationFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    D3D_FEATURE_LEVEL flOut;
    static const D3D_FEATURE_LEVEL flvl[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
    
    BOOL bNeedRefDevice = FALSE;
    if ( !bForceRef )
    {
        hr = Dynamic_D3D11CreateDevice( NULL,                        // Use default graphics card
                                        D3D_DRIVER_TYPE_HARDWARE,    // Try to create a hardware accelerated device
                                        NULL,                        // Do not use external software rasterizer module
                                        uCreationFlags,              // Device creation flags
                                        flvl,
                                        sizeof(flvl) / sizeof(D3D_FEATURE_LEVEL),
                                        D3D11_SDK_VERSION,           // SDK version
                                        ppDeviceOut,                 // Device out
                                        &flOut,                      // Actual feature level created
                                        ppContextOut );              // Context out
        
        if ( SUCCEEDED( hr ) )
        {
            // A hardware accelerated device has been created, so check for Compute Shader support

            // If we have a device >= D3D_FEATURE_LEVEL_11_0 created, full CS5.0 support is guaranteed, no need for further checks
            if ( flOut < D3D_FEATURE_LEVEL_11_0 )            
            {
#ifdef TEST_DOUBLE
                bNeedRefDevice = TRUE;
                printf( "No hardware Compute Shader 5.0 capable device found (required for doubles), trying to create ref device.\n" );
#else
                // Otherwise, we need further check whether this device support CS4.x (Compute on 10)
                D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS hwopts;
                (*ppDeviceOut)->CheckFeatureSupport( D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS, &hwopts, sizeof(hwopts) );
                if ( !hwopts.ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x )
                {
                    bNeedRefDevice = TRUE;
                    printf( "No hardware Compute Shader capable device found, trying to create ref device.\n" );
                }
#endif
            }

#ifdef TEST_DOUBLE
            else
            {
                // Double-precision support is an optional feature of CS 5.0
                D3D11_FEATURE_DATA_DOUBLES hwopts;
                (*ppDeviceOut)->CheckFeatureSupport( D3D11_FEATURE_DOUBLES, &hwopts, sizeof(hwopts) );
                if ( !hwopts.DoublePrecisionFloatShaderOps )
                {
                    bNeedRefDevice = TRUE;
                    printf( "No hardware double-precision capable device found, trying to create ref device.\n" );
                }
            }
#endif
        }
    }
    
    if ( bForceRef || FAILED(hr) || bNeedRefDevice )
    {
        // Either because of failure on creating a hardware device or hardware lacking CS capability, we create a ref device here

        SAFE_RELEASE( *ppDeviceOut );
        SAFE_RELEASE( *ppContextOut );
        
        hr = Dynamic_D3D11CreateDevice( NULL,                        // Use default graphics card
                                        D3D_DRIVER_TYPE_REFERENCE,   // Try to create a hardware accelerated device
                                        NULL,                        // Do not use external software rasterizer module
                                        uCreationFlags,              // Device creation flags
                                        flvl,
                                        sizeof(flvl) / sizeof(D3D_FEATURE_LEVEL),
                                        D3D11_SDK_VERSION,           // SDK version
                                        ppDeviceOut,                 // Device out
                                        &flOut,                      // Actual feature level created
                                        ppContextOut );              // Context out
        if ( FAILED(hr) )
        {
            printf( "Reference rasterizer device create failure\n" );
            return hr;
        }
    }

    return hr;
}

}