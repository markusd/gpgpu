//---------------------------------------------------------------------

cbuffer cbWaveDesc : register( b0 )
{
	float dt;
	int count;
	float2 offset;
};

cbuffer cbWaveData : register( b1 )
{
	float4 WavePos[32];
}


struct VS_INPUT
{
    float4 Pos : POSITION;
    float2 Tex : TEXCOORD0;
};

struct PS_INPUT
{
    float4 Pos : SV_POSITION;
    float2 Tex : TEXCOORD0;
};


//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------
PS_INPUT VS( VS_INPUT input )
{
	PS_INPUT output = (PS_INPUT)0;
	output.Pos = input.Pos;
	output.Tex = input.Tex;
	return output;
}


//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------
float4 PS( PS_INPUT input ) : SV_Target
{
	float _dt = dt * 0.333333333f;

	float amp = 0.0f;
	float waves = 2.5f / (float)count;

	for (int i = 0; i < count; ++i) {
		float2 dp = input.Tex - float2(WavePos[i].xy);
		amp += waves * sin(6.2831f * (_dt - length(dp) * 5.0f));
	}

	float4 color = float4(0.0f, 0.0f, 0.0f, 0.0f);

	amp = 2.0f * (amp < 0.0f ? -amp : amp);

	if (amp <= 1.0f) {
		color.r = 1.0f;
		color.g = amp;
	} else if (amp <= 2.0f) {
		color.r = (1.96f - amp);
		color.g = 1.0f;
	} else if (amp <= 3.0f) {
		color.g = 1.0f;
		color.b = (amp - 2.0f);
	} else if (amp <= 4.0f) {
		color.g = (4.0f - amp);
		color.b = 1.0f;
	} else {
		color.r = (amp - 4.01f);
		color.b = 1.0f;
	}

	return float4(1.0f, 1.0f, 1.0f, 1.0f) - color;
}
