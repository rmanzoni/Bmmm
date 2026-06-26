'''
https://docs.google.com/presentation/d/1EHxQcWzw8IxPgCn8hm1prwSP-EktFtiuaEzH8WkQNVY/edit?slide=id.g36971dee1d1_0_0#slide=id.g36971dee1d1_0_0


/ParkingDoubleMuonLowMass0/Run2024C-MINIv6NANOv15-v1/MINIAOD
/ParkingDoubleMuonLowMass0/Run2024D-MINIv6NANOv15-v1/MINIAOD
/ParkingDoubleMuonLowMass0/Run2024E-MINIv6NANOv15-v1/MINIAOD
/ParkingDoubleMuonLowMass0/Run2024F-MINIv6NANOv15-v3/MINIAOD
/ParkingDoubleMuonLowMass0/Run2024G-MINIv6NANOv15-v3/MINIAOD
/ParkingDoubleMuonLowMass0/Run2024H-MINIv6NANOv15-v3/MINIAOD
/ParkingDoubleMuonLowMass0/Run2024I-MINIv6NANOv15-v3/MINIAOD
/ParkingDoubleMuonLowMass0/Run2024I-MINIv6NANOv15_v2-v2/MINIAOD

'''


import subprocess

eras = [
    ('C', '-v1'), 
    ('D', '-v1'), 
    ('E', '-v1'), 
    ('F', '-v3'), 
    ('G', '-v3'), 
    ('H', '-v3'), 
    ('I', '-v3'), 
    ('I', '_v2-v2'), 
]

for (iera, iversion) in eras:
    for part in range(8):
        dataset = f'/ParkingDoubleMuonLowMass{part}/Run2024{iera}-MINIv6NANOv15{iversion}/MINIAOD'
        # Build output filename: strip leading slash, replace remaining slashes with dashes
        outfile = dataset.lstrip('/').replace('/', '-') + '.txt'
        print(f'Querying: {dataset}')
        with open(outfile, 'w') as f:
            result = subprocess.run(
                ['dasgoclient', f'-query=file dataset={dataset}'],
                stdout=f,
                stderr=subprocess.PIPE,
            )
        if result.returncode != 0:
            print(f'  WARNING: dasgoclient failed for {dataset}')
            print(f'  stderr: {result.stderr.decode().strip()}')
        else:
            print(f'  Saved -> {outfile}')
