clear all
swathFolder={'Hydro/NS','Hydro/MS','Hydro/HS'};
com='comfile2017.txt';
nb=1;
for i=1:length(swathFolder)
    mkdir(swathFolder{i});
end


    % LaTop---- -----------
    %          |           |
    %          |           |
    %          |           |
    %          |           |
    % LaBottom- -----------
    %          |           |
    %       LoLeft      LoRight
    
 %Boundries=[64,168,44,132]; %boundries like this [LaTop,LoRight,LaBottom,LoLeft], clockwise
 %as follows: CenterPoint=[LoMidPoint,LaMidPoint]
 %CenterPoint=[0.5*(Boundries(2)+Boundries(4)),0.5*(Boundries(1)+Boundries(3))]
 
 
 LaBottom(1) = 44;
 LaTop(1) = 64;
 La0(1) = 0.5*(LaBottom(1)+LaTop(1));
 
 LoLeft(1) = 132;
 LoRight(1) = 168;
 Lo0(1) = 0.5*(LoLeft(1)+LoRight(1));

 fid = fopen(com,'rt');
 sh=76; 
 day='01';
 %Counter of days done
 numberOfDays=0;
 fdone=0;
%  k=1;
%     hydroPic=ls('Hydrometer_Okhotsk/2016-2018/2016');
%     Amount=size(hydroPic);
%         for i=4:Amount(1)
%             wantedMonthDay{k}=hydroPic(i,3:6); %dates of pictures we have for this year
%             k=k+1;
%         end
        
 i=0;k=0;

 name_start = '2A.';
while ~feof(fid)
    L = fgets(fid);
    fname_start=strfind(L,name_start);
    fstrt = fname_start(end);
    
    year = L(fstrt+22:fstrt+25);
    month = L(fstrt+26:fstrt+27);
    p_day=day;
    day = L(fstrt+28:fstrt+29);
    orbit=L(fstrt+31:fstrt+37);
    whatK=L(fstrt+7:fstrt+8);
    %orbit = strcat('S',L(sh+6:sh+9),'00');
%     NeededDate = any(strcmp(wantedMonthDay,strcat(month,day)));
    %Counter of days done
        if ~strcmp(p_day,day)
            numberOfDays=numberOfDays+1;
           displayText=strcat('Days done: ',num2str(numberOfDays));
           disp(displayText) 
        end
%         if NeededDate  %only sort if matches date
            fn = sscanf(L,'%s');
            fileinfo = hdf5info(fn);
            % Groups(1) - HS, Ka-band; Groups(3) - NS, Ku-band
           
            if strcmp(whatK,'Ku')
                n=1;
                folderPath=swathFolder{1};
                kmPerSquare=5;
            elseif strcmp(whatK,'Ka')
                n=2; %1 for HS and 2 for MS
                folderPath=swathFolder{2};
                kmPerSquare=5;
            end
            
            LaKu = hdf5read(fileinfo.GroupHierarchy.Groups(n).Datasets(1));
            LoKu = hdf5read(fileinfo.GroupHierarchy.Groups(n).Datasets(2));
            sigmaKu = hdf5read(fileinfo.GroupHierarchy.Groups(n).Groups(9).Datasets(5));
            secofdayKu = hdf5read(fileinfo.GroupHierarchy.Groups(n).Groups(8).Datasets(8));
            preciprateKu = hdf5read(fileinfo.GroupHierarchy.Groups(n).Groups(6).Datasets(10));
            IncAngleKu = hdf5read(fileinfo.GroupHierarchy.Groups(n).Groups(5).Datasets(11)); % ????????? ?????? ?????????? :)
%             IncAngleKu = hdf5read(fileinfo.GroupHierarchy.Groups(n).Groups(5).Datasets(10));
            sizeKu = size(LaKu);
            Lsw = sizeKu(1)*kmPerSquare;
            
            [folderStatus,folderMessage] = mkdir(strcat(folderPath,'\m',month,'y',year));
            filePath=strcat(folderPath,'\m',month,'y',year,'\d',day,'m',month,'y',year,orbit);
            
            disp('Counting distances')
            npolosy = 1;
            [d21,j11]= smallestDistance(LaKu,LoKu,La0,Lo0,sizeKu,npolosy);
            npolosy = sizeKu(1);
            [d22,j22]= smallestDistance(LaKu,LoKu,La0,Lo0,sizeKu,npolosy);
            disp('Finished counting distances')
            k1=sortAll(LaKu,LoKu,LaBottom,LoLeft,LaTop,LoRight,d21,d22,j11,j22,filePath,whatK,Lsw,kmPerSquare,sizeKu,IncAngleKu,sigmaKu,preciprateKu,secofdayKu);
        
%         end
end

disp('Done!')
 