function format_plot(a,x)
a.FontSize = 16;
xlim(a,x([1,end]))
a.XTick = x; a.XTickLabel = x;
a.XMinorTick = 'off';
xlabel('# Ranks')
a.YTickLabel = a.YTick;
end