%% Max vs Min
clear
data = load('exp_bgq64.csv');
data = sortrows(data,1);
hi = data(:,5:9);

delta = (data(:,5:9) - data(:,10:14));
delta = delta(:,2:end)./hi(:,2:end);
disp(sum(delta<0.025))
disp(sum(delta<0.1))

%% Blocking vs Non-blocking
data = load('exp_kratos.csv');
data = sortrows(data,[1,2]);
t_block = reshape(data(:,4),4,4);
data = load('exp_kratos_async.csv');
data = sortrows(data,[1,2]);
t_async = reshape(data(:,4),4,4);
% fsizes = [32; 128; 512; 2048]; ranks = [1 2 4 8];
pd = (t_async-t_block)./t_block; % percent difference
disp(pd)

%% Distributed vs Individual File I/O
clear
data = load('exp_bgq64.csv');
data = sortrows(data,1);
ranks = 2.^(0:11);
nr = data(:,1); % number of ranks
nn = data(:,2); % number of neurons
ttot = data(:,3); % also (:,5) ***
c = 'bgrcmyk';

fsizes = [32 128 512 2048 8192];
figure()
for i = 1:numel(fsizes)
    mask = (nn==fsizes(i));
    x = nr(mask);
    z = ttot(mask);
    semilogx(x,z,c(i),'LineWidth',2)
    hold on
end

data = load('exp_bgq64_io.csv');
data = sortrows(data,1);
nr = data(:,1); % number of ranks
nn = data(:,2); % number of neurons
ttot = data(:,3); % also (:,5) ***
for i = 1:numel(fsizes)
    mask = (nn==fsizes(i));
    x = nr(mask);
    z = ttot(mask);
    semilogx(x,z,[c(i) ':'],'LineWidth',2)
    hold on
end

legend(arrayfun(@(x) sprintf('%d',x),fsizes,'uni',false))
format_plot(gca,ranks)
ylabel('Time  -  Seconds')
hold off

%% Compare 64, 32, 16, and 8 ranks/node
clear
ranks = 2.^(0:11);
c = 'bgrcmyk';
ranks_per_node = [64,32,16,8];

figure()
h = zeros(1,4);
for j = 1:numel(ranks_per_node)
    data = load(sprintf('exp_bgq%d.csv',ranks_per_node(j)));
    data = sortrows(data,1);
    nr = data(:,1); % number of ranks
    nn = data(:,2); % number of neurons
    ttot = data(:,4);
    
    fsizes = [32 128 512 2048 8192];
    for i = 1:numel(fsizes)
        mask = (nn==fsizes(i));
        x = nr(mask);
        z = ttot(mask);
        htmp = loglog(x,z,c(j),'LineWidth',2);
        if (i == 1)
            h(j) = htmp;
        end
        hold on
    end
end

legend(h, arrayfun(@(x) sprintf('%02d Ranks/Node',x), ranks_per_node,'uni', false))
format_plot(gca,ranks)
ylabel('Time  -  Seconds')
hold off

% %% 64 ranks/node vs 16 ranks/node
% clear
% data = load('exp_bgq16.csv');
% data = sortrows(data,1);
% nr = data(:,1); % number of ranks
% nn = data(:,2); % number of neurons
% ttot = data(:,4);
% ranks = 2.^(0:11);
% c = 'bgrcmyk';
% 
% fsizes = [32 128 512 2048 8192];
% figure()
% for i = 1:numel(fsizes)
%     mask = (nn==fsizes(i));
%     x = nr(mask);
%     z = ttot(mask);
%     loglog(x,z,c(i),'LineWidth',2)
%     hold on
% end
% 
% data = load('exp_bgq32.csv');
% data = sortrows(data,1);
% nr = data(:,1); % number of ranks
% nn = data(:,2); % number of neurons
% ttot = data(:,4);
% for i = 1:numel(fsizes)
%     mask = (nn==fsizes(i));
%     x = nr(mask);
%     z = ttot(mask);
%     loglog(x,z,'k:','LineWidth',2)
%     hold on
% end
% 
% legend([arrayfun(@(x) sprintf('%d',x),fsizes,'uni',false),{'Distributed'}])
% format_plot(gca,ranks)
% ylabel('Time  -  Seconds')
% hold off
