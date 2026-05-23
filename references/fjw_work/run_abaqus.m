if exist('vert.lck','file')
delete('vert.lck');
else
end
system(['abaqus ','job','=vert',' cpus','=8']);
pause(5);
while(exist('vert.lck','file'))
    pause(0.5);
end