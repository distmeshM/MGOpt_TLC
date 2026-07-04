cd mex
buildDir = fullfile(pwd, 'build');
if exist(buildDir, 'dir')
    rmdir(buildDir, 's');
end
mkdir(buildDir)
cd(buildDir)
system('cmake .. -DCMAKE_BUILD_TYPE=Release');
system('cmake --build . --config Release');
cd ../../surface_multigrid/mex
buildDir = fullfile(pwd, 'build');
if exist(buildDir, 'dir')
    rmdir(buildDir, 's');
end
mkdir(buildDir)
cd(buildDir)
system('cmake .. -DCMAKE_BUILD_TYPE=Release');
system('cmake --build . --config Release');