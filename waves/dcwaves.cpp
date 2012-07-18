#include <waves.hpp>

#include <windows.h>
#include <D3D/d3d11.h>
#include <D3D/d3dx11.h>
#include <D3D/d3dcompiler.h>
#include <m3d/m3d.hpp>
#include <util/tostring.hpp>
#include <util/clock.hpp>

#include <stdio.h>
#include <io.h>
#include <fcntl.h>

#include <iostream>
#include <fstream>

using namespace m3d;

#define IDI_TUTORIAL1           107

//--------------------------------------------------------------------------------------
// Structures
//--------------------------------------------------------------------------------------
struct SimpleVertex
{
    Vec3f Pos;
	Vec2f Tex;
};

struct CBWaveDesc
{
	float dt;
	int count;
	Vec2f offset;
};


struct CBWaveData
{
	Vec4f WavePos[32];
};


//--------------------------------------------------------------------------------------
// Global Variables
//--------------------------------------------------------------------------------------
HINSTANCE               g_hInst = NULL;
HWND                    g_hWnd = NULL;
D3D_DRIVER_TYPE         g_driverType = D3D_DRIVER_TYPE_NULL;
D3D_FEATURE_LEVEL       g_featureLevel = D3D_FEATURE_LEVEL_11_0;
ID3D11Device*           g_pd3dDevice = NULL;
ID3D11DeviceContext*    g_pImmediateContext = NULL;
IDXGISwapChain*         g_pSwapChain = NULL;
ID3D11RenderTargetView* g_pRenderTargetView = NULL;
ID3D11VertexShader*     g_pVertexShader = NULL;
ID3D11PixelShader*      g_pPixelShader = NULL;
ID3D11InputLayout*      g_pVertexLayout = NULL;
ID3D11Buffer*           g_pVertexBuffer = NULL;
ID3D11Buffer*           g_pCBWaveDesc = NULL;
ID3D11Buffer*           g_pCBWaveData = NULL;
ID3D11ComputeShader*	g_pCSWaves = NULL;
ID3D11Texture2D*		g_pTexture = NULL;
ID3D11UnorderedAccessView* g_pTextureView = NULL;
ID3D11ShaderResourceView* g_pTextureResourceView = NULL;
ID3D11SamplerState* g_pSamplerLinear = NULL;

util::Clock g_clock;
CBWaveDesc g_cbWaveDesc;
CBWaveData g_cbWaveData;
float* texData = NULL;

std::ofstream g_logfile;


//--------------------------------------------------------------------------------------
// Forward declarations
//--------------------------------------------------------------------------------------
HRESULT InitWindow( HINSTANCE hInstance, int nCmdShow );
HRESULT InitDevice();
void CleanupDevice();
LRESULT CALLBACK    WndProc( HWND, UINT, WPARAM, LPARAM );
void Render();


//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
#ifdef USE_DIRECT_COMPUTE
int WINAPI wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow )
{
    UNREFERENCED_PARAMETER( hPrevInstance );
    UNREFERENCED_PARAMETER( lpCmdLine );

	    AllocConsole();

    HANDLE handle_out = GetStdHandle(STD_OUTPUT_HANDLE);
    int hCrt = _open_osfhandle((long) handle_out, _O_TEXT);
    FILE* hf_out = _fdopen(hCrt, "w");
    setvbuf(hf_out, NULL, _IONBF, 1);
    *stdout = *hf_out;

    HANDLE handle_in = GetStdHandle(STD_INPUT_HANDLE);
    hCrt = _open_osfhandle((long) handle_in, _O_TEXT);
    FILE* hf_in = _fdopen(hCrt, "r");
    setvbuf(hf_in, NULL, _IONBF, 128);
    *stdin = *hf_in;

	
	//int w, h, c;
	//unsigned char* in_data = stbi_load("tex.bmp", &w, &h, &c, STBI_rgb_alpha);

	//if (!in_data) {
	//	std::cout << "Error: Could not read input file" << std::endl;
	//	return 1;
	//}

	int w = WIDTH;
	int h = HEIGHT;
	texData = new float[WIDTH * WIDTH*4];

	for (int x = 0; x < w; ++x) {
		for (int y = 0; y < h; ++y) {

			
			//unsigned char R = (x / (float)WIDTH * 255);
			//unsigned char G = (y / (float)HEIGHT * 255);
			//unsigned char B = R + G;
			//unsigned char A = 255;

			//texData[x+y*w] = x + y;
			texData[(x+y*w)*4+0] = x / (float)WIDTH;//in_data[(x+y*w)*4+0] / 255.0f;
			texData[(x+y*w)*4+1] = y / (float)HEIGHT;//in_data[(x+y*w)*4+1] / 255.0f;
			texData[(x+y*w)*4+2] = texData[(x+y*w)*4+0] + texData[(x+y*w)*4+1];//in_data[(x+y*w)*4+2] / 255.0f;
			texData[(x+y*w)*4+3] = 1.0f;//in_data[(x+y*w)*4+3] / 255.0f;
		}
	}


    if( FAILED( InitWindow( hInstance, nCmdShow ) ) )
        return 0;

    if( FAILED( InitDevice() ) )
    {
        CleanupDevice();
        return 0;
    }

	g_logfile.open("log.csv");
	g_logfile << "fps\n";
	
	g_clock.reset();

    // Main message loop
    MSG msg = {0};
    while( WM_QUIT != msg.message )
    {
        if( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) )
        {
            TranslateMessage( &msg );
            DispatchMessage( &msg );
        }
        else
        {
            Render();
        }
    }

	g_logfile.close();
    CleanupDevice();

    return ( int )msg.wParam;
}
#endif


//--------------------------------------------------------------------------------------
// Register class and create window
//--------------------------------------------------------------------------------------
HRESULT InitWindow( HINSTANCE hInstance, int nCmdShow )
{
    // Register class
    WNDCLASSEX wcex;
    wcex.cbSize = sizeof( WNDCLASSEX );
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIcon( hInstance, ( LPCTSTR )IDI_TUTORIAL1 );
    wcex.hCursor = LoadCursor( NULL, IDC_ARROW );
    wcex.hbrBackground = ( HBRUSH )( COLOR_WINDOW + 1 );
    wcex.lpszMenuName = NULL;
    wcex.lpszClassName = L"WavesWindowClass";
    wcex.hIconSm = LoadIcon( wcex.hInstance, ( LPCTSTR )IDI_TUTORIAL1 );
    if( !RegisterClassEx( &wcex ) )
        return E_FAIL;

    // Create window
    g_hInst = hInstance;
    RECT rc = { 0, 0, WIDTH, HEIGHT };
    AdjustWindowRect( &rc, WS_OVERLAPPEDWINDOW, FALSE );
    g_hWnd = CreateWindow( L"WavesWindowClass", L"Waves - Direct Compute",
                           WS_OVERLAPPEDWINDOW,
                           CW_USEDEFAULT, CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top, NULL, NULL, hInstance,
                           NULL );
    if( !g_hWnd )
        return E_FAIL;

    ShowWindow( g_hWnd, nCmdShow );

    return S_OK;
}


//--------------------------------------------------------------------------------------
// Helper for compiling shaders with D3DX11
//--------------------------------------------------------------------------------------
HRESULT CompileShaderFromFile( WCHAR* szFileName, LPCSTR szEntryPoint, LPCSTR szShaderModel, ID3DBlob** ppBlobOut )
{
    HRESULT hr = S_OK;

    DWORD dwShaderFlags = 0;// D3DCOMPILE_ENABLE_STRICTNESS;
#if defined( DEBUG ) || defined( _DEBUG )
    // Set the D3DCOMPILE_DEBUG flag to embed debug information in the shaders.
    // Setting this flag improves the shader debugging experience, but still allows 
    // the shaders to be optimized and to run exactly the way they will run in 
    // the release configuration of this program.
    dwShaderFlags |= D3DCOMPILE_DEBUG;
#endif

    ID3DBlob* pErrorBlob;
    hr = D3DX11CompileFromFile( szFileName, NULL, NULL, szEntryPoint, szShaderModel, 
        dwShaderFlags, 0, NULL, ppBlobOut, &pErrorBlob, NULL );

    if( FAILED(hr) )
    {
        if( pErrorBlob != NULL )
			MessageBoxA( NULL,
                   (char*)pErrorBlob->GetBufferPointer() , "Error", MB_OK );
            //OutputDebugStringA( (char*)pErrorBlob->GetBufferPointer() );
        if( pErrorBlob ) pErrorBlob->Release();
        return hr;
    }
    if( pErrorBlob ) pErrorBlob->Release();

    return S_OK;
}


//--------------------------------------------------------------------------------------
// Create Direct3D device and swap chain
//--------------------------------------------------------------------------------------
HRESULT InitDevice()
{
    HRESULT hr = S_OK;

    RECT rc;
    GetClientRect( g_hWnd, &rc );
    UINT width = rc.right - rc.left;
    UINT height = rc.bottom - rc.top;

    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_DRIVER_TYPE driverTypes[] =
    {
        D3D_DRIVER_TYPE_HARDWARE,
        D3D_DRIVER_TYPE_WARP,
        D3D_DRIVER_TYPE_REFERENCE,
    };
    UINT numDriverTypes = ARRAYSIZE( driverTypes );

    D3D_FEATURE_LEVEL featureLevels[] =
    {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    };
	UINT numFeatureLevels = ARRAYSIZE( featureLevels );

    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory( &sd, sizeof( sd ) );
    sd.BufferCount = 1;
    sd.BufferDesc.Width = width;
    sd.BufferDesc.Height = height;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = g_hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;

    for( UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes; driverTypeIndex++ )
    {
        g_driverType = driverTypes[driverTypeIndex];
        hr = D3D11CreateDeviceAndSwapChain( NULL, g_driverType, NULL, createDeviceFlags, featureLevels, numFeatureLevels,
                                            D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &g_featureLevel, &g_pImmediateContext );
        if( SUCCEEDED( hr ) )
            break;
    }
    if( FAILED( hr ) )
        return hr;

    // Create a render target view
    ID3D11Texture2D* pBackBuffer = NULL;
    hr = g_pSwapChain->GetBuffer( 0, __uuidof( ID3D11Texture2D ), ( LPVOID* )&pBackBuffer );
    if( FAILED( hr ) )
        return hr;

    hr = g_pd3dDevice->CreateRenderTargetView( pBackBuffer, NULL, &g_pRenderTargetView );
    pBackBuffer->Release();
    if( FAILED( hr ) )
        return hr;

    g_pImmediateContext->OMSetRenderTargets( 1, &g_pRenderTargetView, NULL );

    // Setup the viewport
    D3D11_VIEWPORT vp;
    vp.Width = (FLOAT)width;
    vp.Height = (FLOAT)height;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    g_pImmediateContext->RSSetViewports( 1, &vp );

	// Compile the compute shader
	ID3DBlob* pCSBlob = NULL;
	hr = CompileShaderFromFile(L"dcwaves.fx", "CSWaves", "cs_5_0", &pCSBlob);
    if( FAILED( hr ) )
    {
        MessageBox( NULL,
                    L"The CSFX file cannot be compiled.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK );
        return hr;
    }

	hr = g_pd3dDevice->CreateComputeShader( pCSBlob->GetBufferPointer(), pCSBlob->GetBufferSize(), NULL, &g_pCSWaves );
    if( FAILED( hr ) )
        return hr;
	pCSBlob->Release();

	// Create Texture
	D3D11_TEXTURE2D_DESC textureDesc;
	ZeroMemory(&textureDesc, sizeof(textureDesc));
	textureDesc.Width = WIDTH;
	textureDesc.Height = HEIGHT;
	textureDesc.MipLevels = 1;
	textureDesc.ArraySize = 1;
	textureDesc.SampleDesc.Count = 1;
	textureDesc.SampleDesc.Quality = 0;
	textureDesc.Usage = D3D11_USAGE_DEFAULT;
	textureDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
	textureDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;//DXGI_FORMAT_R8G8B8A8_UNORM
	g_pd3dDevice->CreateTexture2D(&textureDesc, NULL, &g_pTexture);

	g_pImmediateContext->UpdateSubresource(g_pTexture, 0, NULL, texData, 0, 0);

	D3D11_UNORDERED_ACCESS_VIEW_DESC viewDescUAV;
	ZeroMemory(&viewDescUAV, sizeof(viewDescUAV));
	viewDescUAV.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	viewDescUAV.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
	viewDescUAV.Texture2D.MipSlice = 0;
	g_pd3dDevice->CreateUnorderedAccessView(g_pTexture, &viewDescUAV, &g_pTextureView);
	    if( FAILED( hr ) )
        return hr;

	
	D3D11_SHADER_RESOURCE_VIEW_DESC viewDescRes;
	ZeroMemory(&viewDescRes, sizeof(viewDescRes));
	viewDescRes.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	viewDescRes.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	viewDescRes.Texture2D.MipLevels = 1;
	viewDescRes.Texture2D.MostDetailedMip = 0;
	hr = g_pd3dDevice->CreateShaderResourceView(g_pTexture, &viewDescRes, &g_pTextureResourceView);

	    if( FAILED( hr ) )
        return hr;

	    // Load the Texture
/*
		D3DX11_IMAGE_LOAD_INFO loadInfo;
ZeroMemory( &loadInfo, sizeof(D3DX11_IMAGE_LOAD_INFO) );
loadInfo.BindFlags = D3D11_BIND_SHADER_RESOURCE;
loadInfo.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;

hr = D3DX11CreateShaderResourceViewFromFile( g_pd3dDevice, L"seafloor.dds", &loadInfo, NULL, &g_pTextureResourceView, NULL );
    if( FAILED( hr ) )
        return hr;

*/


    // Compile the vertex shader
    ID3DBlob* pVSBlob = NULL;
#ifdef USE_D3D11
    hr = CompileShaderFromFile( L"waves.fx", "VS", "vs_5_0", &pVSBlob );
#else
	hr = CompileShaderFromFile( L"dcwavesview.fx", "VS", "vs_5_0", &pVSBlob );
#endif
    if( FAILED( hr ) )
    {
        MessageBox( NULL,
                    L"The VIEWFX file cannot be compiled.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK );
        return hr;
    }

	// Create the vertex shader
	hr = g_pd3dDevice->CreateVertexShader( pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), NULL, &g_pVertexShader );
	if( FAILED( hr ) )
	{	
		pVSBlob->Release();
        return hr;
	}

    // Define the input layout
    D3D11_INPUT_ELEMENT_DESC layout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
	UINT numElements = ARRAYSIZE( layout );

    // Create the input layout
	hr = g_pd3dDevice->CreateInputLayout( layout, numElements, pVSBlob->GetBufferPointer(),
                                          pVSBlob->GetBufferSize(), &g_pVertexLayout );
	pVSBlob->Release();
	if( FAILED( hr ) )
        return hr;

    // Set the input layout
    g_pImmediateContext->IASetInputLayout( g_pVertexLayout );

	// Compile the pixel shader
	ID3DBlob* pPSBlob = NULL;
#ifdef USE_D3D11
	hr = CompileShaderFromFile( L"waves.fx", "PS", "ps_5_0", &pPSBlob );
#else
	hr = CompileShaderFromFile( L"dcwavesview.fx", "PS", "ps_5_0", &pPSBlob );
#endif

    if( FAILED( hr ) )
    {
        MessageBox( NULL,
                    L"The VIEWPSFX file cannot be compiled.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK );
        return hr;
    }

	// Create the pixel shader
	hr = g_pd3dDevice->CreatePixelShader( pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), NULL, &g_pPixelShader );
	pPSBlob->Release();
    if( FAILED( hr ) )
        return hr;

    // Create vertex buffer
    SimpleVertex vertices[] =
    {
		{ Vec3f( -1.0f, -1.0f, 0.5f ), Vec2f(0.0f, 0.0f) },
        { Vec3f( -1.0f,  1.0f, 0.5f ), Vec2f(0.0f, 1.0f) },
        { Vec3f(  1.0f, -1.0f, 0.5f ), Vec2f(1.0f, 0.0f) },
        { Vec3f(  1.0f,  1.0f, 0.5f ), Vec2f(1.0f, 1.0f) },
    };
    D3D11_BUFFER_DESC bd;
	ZeroMemory( &bd, sizeof(bd) );
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.ByteWidth = sizeof( SimpleVertex ) * 4;
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = 0;
    D3D11_SUBRESOURCE_DATA InitData;
	ZeroMemory( &InitData, sizeof(InitData) );
    InitData.pSysMem = vertices;
    hr = g_pd3dDevice->CreateBuffer( &bd, &InitData, &g_pVertexBuffer );
    if( FAILED( hr ) )
        return hr;

    // Set vertex buffer
    UINT stride = sizeof( SimpleVertex );
    UINT offset = 0;
    g_pImmediateContext->IASetVertexBuffers( 0, 1, &g_pVertexBuffer, &stride, &offset );

    // Set primitive topology
    g_pImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );


	// Create the constant buffers
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.ByteWidth = sizeof(CBWaveDesc);
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	//bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.CPUAccessFlags = 0;
    hr = g_pd3dDevice->CreateBuffer( &bd, NULL, &g_pCBWaveDesc );
    if( FAILED( hr ) )
        return hr;

	// Create the constant buffers
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.ByteWidth = sizeof(CBWaveData);
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	//bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.CPUAccessFlags = 0;
    hr = g_pd3dDevice->CreateBuffer( &bd, NULL, &g_pCBWaveData );
    if( FAILED( hr ) )
        return hr;
    
	// Create the sample state
    D3D11_SAMPLER_DESC sampDesc;
    ZeroMemory( &sampDesc, sizeof(sampDesc) );
	sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
    sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
    sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    sampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampDesc.MinLOD = 0;
    sampDesc.MaxLOD = D3D11_FLOAT32_MAX;
    hr = g_pd3dDevice->CreateSamplerState( &sampDesc, &g_pSamplerLinear );
    if( FAILED( hr ) )
        return hr;

		
	
    g_cbWaveDesc.count = 0;
	g_cbWaveDesc.dt = g_clock.get();
    g_pImmediateContext->UpdateSubresource( g_pCBWaveDesc, 0, NULL, &g_cbWaveDesc, 0, 0 );


    return S_OK;
}


//--------------------------------------------------------------------------------------
// Clean up the objects we've created
//--------------------------------------------------------------------------------------
void CleanupDevice()
{
    if( g_pImmediateContext ) g_pImmediateContext->ClearState();

    if( g_pVertexBuffer ) g_pVertexBuffer->Release();
	if( g_pCBWaveDesc ) g_pCBWaveDesc->Release();
	if( g_pCBWaveData ) g_pCBWaveData->Release();
    if( g_pVertexLayout ) g_pVertexLayout->Release();
    if( g_pVertexShader ) g_pVertexShader->Release();
    if( g_pPixelShader ) g_pPixelShader->Release(); 
    if( g_pRenderTargetView ) g_pRenderTargetView->Release();
    if( g_pSwapChain ) g_pSwapChain->Release();
    if( g_pImmediateContext ) g_pImmediateContext->Release();
    if( g_pd3dDevice ) g_pd3dDevice->Release();
}


//--------------------------------------------------------------------------------------
// Called every time the application receives a message
//--------------------------------------------------------------------------------------
LRESULT CALLBACK WndProc( HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam )
{
    PAINTSTRUCT ps;
    HDC hdc;

    switch( message )
    {
        case WM_PAINT:
            hdc = BeginPaint( hWnd, &ps );
            EndPaint( hWnd, &ps );
            break;

        case WM_DESTROY:
            PostQuitMessage( 0 );
            break;

		case WM_LBUTTONUP:
			g_cbWaveData.WavePos[g_cbWaveDesc.count++] = Vec4f(LOWORD(lParam) / (float)WIDTH, ((float)HEIGHT - HIWORD(lParam)) / (float)HEIGHT, 0.0f, 0.0f);
			g_pImmediateContext->UpdateSubresource( g_pCBWaveData, 0, NULL, &g_cbWaveData, 0, 0 );
			break;

        default:
            return DefWindowProc( hWnd, message, wParam, lParam );
    }

    return 0;
}


//--------------------------------------------------------------------------------------
// Render a frame
//--------------------------------------------------------------------------------------
void Render()
{
	static int frames = 0;
	static float fps_time = 0.0f;

	float dt = g_clock.get();

    // Clear the back buffer 
    float ClearColor[4] = { 0.0f, 0.125f, 0.3f, 1.0f }; // red,green,blue,alpha
    g_pImmediateContext->ClearRenderTargetView( g_pRenderTargetView, ClearColor );

	float time = g_clock.get();

	g_cbWaveDesc.dt = g_clock.get();
    g_pImmediateContext->UpdateSubresource( g_pCBWaveDesc, 0, NULL, &g_cbWaveDesc, 0, 0 );
	
	UINT counts = 0;
	g_pImmediateContext->CSSetShader(g_pCSWaves, NULL, 0);
	g_pImmediateContext->CSSetConstantBuffers(0, 1, &g_pCBWaveDesc);
	g_pImmediateContext->CSSetConstantBuffers(1, 1, &g_pCBWaveData);
	g_pImmediateContext->CSSetUnorderedAccessViews(0, 1, &g_pTextureView, 0);
	g_pImmediateContext->Dispatch(WIDTH/16, HEIGHT/16, 1);
	g_pImmediateContext->Flush();

	// Unbind resources for CS
	ID3D11UnorderedAccessView* ppUAViewNULL[1] = { NULL };
	g_pImmediateContext->CSSetUnorderedAccessViews( 0, 1, ppUAViewNULL, 0 );

	float time2 = g_clock.get();
	std::cout << time2-time << "\n";

    // Render a triangle
	g_pImmediateContext->VSSetShader( g_pVertexShader, NULL, 0 );
	g_pImmediateContext->PSSetShader( g_pPixelShader, NULL, 0 );
#ifdef USE_D3D11
	g_pImmediateContext->PSSetConstantBuffers( 0, 1, &g_pCBWaveDesc );
	g_pImmediateContext->PSSetConstantBuffers( 1, 1, &g_pCBWaveData );
#else
	g_pImmediateContext->PSSetShaderResources(0, 1, &g_pTextureResourceView);
	g_pImmediateContext->PSSetSamplers(0, 1, &g_pSamplerLinear);
#endif
    g_pImmediateContext->Draw( 4, 0 );

    // Present the information rendered to the back buffer to the front buffer (the screen)
    g_pSwapChain->Present( 0, 0 );

	
	frames++;
	if (dt - fps_time >= 1.0f) {
		LPCSTR text = util::toString(frames);
		g_logfile << frames << "\n";
		//std::cout << dt << ", " << fps_time << ", " << frames << std::endl;
		
		SetWindowTextA(g_hWnd, text);
		fps_time = dt;
		frames = 0;
	}
}
