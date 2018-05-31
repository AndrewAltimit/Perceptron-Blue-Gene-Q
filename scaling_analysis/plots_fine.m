clear
data = load('exp_bgq64.csv');
data = sortrows(data,1);
ranks = 2.^(0:11);
nr = data(:,1); % number of ranks
nn = data(:,2); % number of neurons
ttot = data(:,4);
hi = data(:,5:9);
c = 'bgrcmyk';

%% Communication time breakdown
measures = {'Sharing','Fwdprop','Bcast','Backprop','Reduce'};
fsizes = [32 128 512 2048 8192];
for i = 1:numel(fsizes)
    figure(i)
    mask = (nn==fsizes(i));
    x = nr(mask);
    Y = hi(mask,:);
    z = ttot(mask);
%     semilogy(x,(Y(:,2)-Y(:,3))./Y(:,2),'b',...
%         x,(Y(:,4)-Y(:,5))./Y(:,4),'g',...
%         x,z(1)./(x.*z),'r','LineWidth',2) % efficiency
    
    loglog(x,Y(:,3),'b:',x,Y(:,5),'g:',x,Y(:,2),'b',x,Y(:,4),'g',x,z,'k','LineWidth',2)
    legend('Bcast','Reduce','Forward','Backward','Overall')
    format_plot(gca,x)
    ylabel('Time  -  Seconds')
end

%% Execution Time
fsizes = [32 128 512 2048 8192];
figure()
for i = 1:numel(fsizes)
    mask = (nn==fsizes(i));
    x = nr(mask);
    z = ttot(mask);
    loglog(x,z,c(i),'LineWidth',2)
    hold on
end
legend(arrayfun(@(x) sprintf('%d',x),fsizes,'uni',false))
format_plot(gca,ranks)
ylabel('Time  -  Seconds')
hold off

%% Speedup
fsizes = [32 128 512 2048 8192];
figure()
ref = 32^2/ttot(nn==32 & nr==1);
for i = 1:numel(fsizes)
    mask = (nn==fsizes(i));
    x = nr(mask);
    z = ttot(mask);
    loglog(x,(fsizes(i)^2./z)/ref,c(i),'LineWidth',2)
    hold on
end
for i = 1:numel(fsizes)-1
    mask = (nn==fsizes(i));
    x = nr(mask);
    z = ttot(mask);
    loglog(x,z(1)./z,[c(i) ':'],'LineWidth',2)
end
legend(arrayfun(@(x) sprintf('%d',x),fsizes,'uni',false))
format_plot(gca,ranks)
ylabel('Speedup') % 1000MACs/Second
hold off

%% Efficiency
fsizes = [32 128 512 2048 8192];
figure()
for i = 1:numel(fsizes)
    mask = (nn==fsizes(i));
    x = nr(mask);
    Y = hi(mask,:);
    semilogx(x,(Y(:,2)+Y(:,4)-Y(:,3)-Y(:,5))./(Y(:,2)+Y(:,4)),c(i),'LineWidth',2) % efficiency
    hold on
end
for i = 1:numel(fsizes)-1
    mask = (nn==fsizes(i));
    x = nr(mask);
    z = ttot(mask);
    semilogx(x,z(1)./(x.*z),[c(i) ':'],'LineWidth',2)
end
legend(arrayfun(@(x) sprintf('%d',x),fsizes,'uni',false))
format_plot(gca,ranks)
ylabel('Efficiency')
hold off

% %% Workload/(Time*Nproc)
% fsizes = [32 128 512 2048 8192];
% figure()
% for i = 1:numel(fsizes)
%     mask = (nn==fsizes(i));
%     x = nr(mask);
%     z = ttot(mask);
%     semilogx(x,(fsizes(i)^2)./(1000*z.*x),c(i),'LineWidth',2)
%     hold on
% end
% format_plot(gca,ranks)
% ylabel('Workload/(Time*Ranks)') % 1000MACs/Second
% hold off

% %% All fine-scale times
% data = load('exp_bgq.csv');
% 
% data = data(data(:,2)==2048,:);
% data = sortrows(data,1);
% 
% nr = data(:,1);
% hi = data(:,5:9);
% lo = data(:,10:14);
% M = cat(1,lo,hi(end:-1:1,:));
% r = cat(1,nr,nr(end:-1:1));
% 
% figure
% semilogy(r,M(:,1),'b',r,M(:,2),'g',r,M(:,3),'r',r,M(:,4),'c',r,M(:,5),'m',nr,data(:,4),'k')
% legend('Sharing','Fwdprop','Bcast','Backprop','Reduce')
% a = gca;
% a.FontSize = 18;
% for k = 1:numel(a.Children)
%     a.Children(k).LineWidth = 2;
% end