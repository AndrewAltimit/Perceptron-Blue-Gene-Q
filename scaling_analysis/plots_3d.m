clear
data = load('exp_bgq64.csv');

f = figure();
data = sortrows(data,[1,2]);

ranks = unique(data(:,1));
neurons = unique(data(:,2));

% Resahep data in a 2D array
chunk = NaN(numel(ranks),numel(neurons));
for i = 1:numel(ranks)
    for j = 1:numel(neurons)
        time = data(data(:,1)==ranks(i) & data(:,2)==neurons(j),4);
        if ~isempty(time)
            chunk(i,j) = log10(time);
        end
    end
end

b = bar3(chunk);

% Color bars based on height/value
for j = 1:length(b)
    zdata = b(j).ZData;
    b(j).CData = zdata;
    b(j).FaceColor = 'interp';
end

% Remove bars for NaN values
for i = 1:numel(ranks)
    for j = 1:numel(neurons)
        if isnan(chunk(i,j))
            b(j).ZData((i-1)*6+1:i*6,:) = NaN;
        end
    end
end

zlim([-1,5])
a = f.Children(1);
a.YDir = 'normal';
a.YTickLabel = ranks; ylabel('# Ranks')
a.XTickLabel = neurons; xlabel('# Neurons/Layer)')
a.ZTickLabel = 10.^(a.ZTick); zlabel('Time (sec)')
% a.FontSize = 18;
title('Overall Training Time')

% print(f,sprintf('read%d.png',nfiles(i)),'-dpng')
