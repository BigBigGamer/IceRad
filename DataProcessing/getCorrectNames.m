clear all
fid=fopen('nasa2018_03.txt','rt');
fileName=fgets(fid);
fileName=fgets(fid);
fileName=fgets(fid);
fileName=fgets(fid);
com=fopen('names2018_03.txt','wt');
while ~feof(fid)
    fileName=fgets(fid);
    %fprintf(com,'%s',fileName(107:end));
    if length(fileName)>10
        fprintf(com,'%s',fileName(114:end));
        %fprintf(com,'%s\n',strcat(fileName(105:115),fileName(134:end)));
    end
end
fclose(fid);
fclose(com);
