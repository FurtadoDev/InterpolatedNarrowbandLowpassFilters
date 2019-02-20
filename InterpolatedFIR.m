% Code illustrating the magnitude response of the band-edge shaping filter, 
the masking filter and the resulting filter. Also contained is the 1) relationship between the percentage 
computation reduction and the transition bandwidth, 2) the relationship between the optimum expansion factor and the 
transition bandwidth, 3) the relationship between the maximum computation reduction and the transition bandwidth. 
Note that the values corresponding to the passband width, the transition width, the expansion factor need to be 
changed when plotting for different values. %

close all
clear all
%% Select the range of alpha you wish to run plots for
   expansion_factor=3;
   M=61;
   wc=0.1*2*pi*expansion_factor;
   alp = M/2;
   n = 0:1:floor(2*alp);
   nfine= linspace(0,ceil(2*alp),300);
   h = (wc/pi)*sinc((wc/pi)*(n-alp)); 
   henv = (wc/pi)*sinc((wc/pi)*(nfine-alp)); 
   a=[1 zeros(1,length(h)-1)];
%Interpolate the impulse response
h_interpolated=upsample(h,expansion_factor);
for r = 1:expansion_factor-1
   h_interpolated(length(h_interpolated))=[];
end
n_interpolated=0:1:length(h_interpolated)-1;
b=[1 zeros(1,length(h_interpolated)-1)];
window_length=((expansion_factor*(M+1-1))+1);
h_interpolated=h_interpolated.*kaiser(window_length,5.65326)';
 
%Mask the interpolated response to attenuate the repetitive bands
wc_ma=0.1*2*pi;
M_ma=32;
n_ma = 0:1:floor(2*(M_ma/2));
h_ma = (wc_ma/pi)*sinc((wc_ma/pi)*(n-(M_ma/2))); 
c=[1 zeros(1,length(h_ma)-1)];
 
% Determine and plot frequency response
   w=linspace(0,2*pi,256);
   H=freqz(h,a,w);
   H_be=freqz(h_interpolated,b,w);
   H_ma=freqz(h_ma,c,w);
   
% Attenuate the repetitive passbands
y=H_ma.*H_be;
%Plot the magnitude response of the prototype, band-edge and masking filter
   figure
   subplot(411), plot(w/(2*pi),abs(H))
   ylabel('|H(w)|'), title('Freq response of the prototype filter')
   subplot(412), plot(w/(2*pi),abs(H_be))
   ylabel('|H(w)|'), title('Freq response of the band-edge shaping filter')
   subplot(413), plot(w/(2*pi),abs(H_ma))
   ylabel('|H(w)|'), title('Freq response of the masking filter')
   subplot(414), plot(w/(2*pi),abs(y))
   ylabel('|H(w)|'), title('Freq response of the resulting filter(band-edge + masking)')
   % Also plot impulse response and zero locations - but omit vry marge magnitude zeros   
   figure
   stem(n,h),
   hold on
   plot(nfine,henv), hold off
   xlabel('n'), ylabel('h(n)'), title('Impulse response')
   subplot(312),stem(n_interpolated,h_interpolated)
   
% Relationship between % computation reduction and transition bandwith
A=60;
fpass=0.1;
M2=2;
M3=3;
M4=4;
 
ftrans1=linspace(0,0.1,21);
max_red=zeros(1,21);
N_tr=((A-8)./(2.285*ftrans1*2*pi))+1;
N_be=((A-8)./(2.285*M2*ftrans1*2*pi))+1;
N_ma=((A-8)./(2.295*(((2*pi)/M2)-((fpass*2*pi)+(ftrans1*2*pi))-(fpass*2*pi))))+1;
comp_reduct2=((N_tr-N_be-N_ma)./N_tr)*100;
 
N_tr=((A-8)./(2.285*ftrans1*2*pi))+1;
N_be=((A-8)./(2.285*M3*ftrans1*2*pi))+1;
N_ma=((A-8)./(2.295*(((2*pi)/M3)-((fpass*2*pi)+(ftrans1*2*pi))-(fpass*2*pi))))+1;
comp_reduct3=((N_tr-N_be-N_ma)./N_tr)*100;
 
N_tr=((A-8)./(2.285*ftrans1*2*pi))+1;
N_be=((A-8)./(2.285*M4*ftrans1*2*pi))+1;
N_ma=((A-8)./(2.295*(((2*pi)/M4)-((fpass*2*pi)+(ftrans1*2*pi))-(fpass*2*pi))))+1;
comp_reduct4=((N_tr-N_be-N_ma)./N_tr)*100;
 
 
for x=1:21
    if(((2*pi)-(M4*ftrans1(x)*2*pi)-(2*M4*fpass*2*pi))<=0)
      comp_reduct4(x)=0;
    elseif(((2*pi)-(M3*ftrans1(x)*2*pi)-(2*M3*fpass*2*pi))<=0)
      comp_reduct3(x)=0;
    elseif(((2*pi)-(M2*ftrans1(x)*2*pi)-(2*M2*fpass*2*pi))<=0)
      comp_reduct2(x)=0;
    end
    max_red(x)=max([comp_reduct4(x),comp_reduct3(x),comp_reduct2(x)]);
end
figure
plot(ftrans1,comp_reduct2,'g')
text(0.09,5,'M=2')
hold on
plot(ftrans1,comp_reduct3,'b')
text(0.04,25,'M=3')
hold on
plot(ftrans1,comp_reduct4,'c')
text(0.02,10,'M=4')
hold on
plot(ftrans1,max_red,'r')
xlabel('Transition Region Bandwidth(X 2*pi radians)')
ylabel('Maximum % Computation Reduction')
axis([0 0.1 0 80])
grid on
%plot(ftrans,comp_reduct1)
 
% Relationship between optimum expansion factor and transition bandwith 
A=60;
ftrans=0:0.01:0.10;
optimum_val=zeros(1,11);
for i = 1:11
N_tr=((A-8)/(2.285*ftrans(i)*2*pi))+1;
N_be=((A-8)/(2.285*M2*ftrans(i)*2*pi))+1;
N_ma=((A-8)/(2.295*(((2*pi)/M2)-((fpass*2*pi)+(ftrans(i)*2*pi))-(fpass*2*pi))))+1;
c2=((N_tr-N_be-N_ma)/N_tr)*100;
 
%N_tr=((A-8)/(2.285*ftrans(i)*2*pi))+1;
N_be=((A-8)/(2.285*M3*ftrans(i)*2*pi))+1;
N_ma=((A-8)/(2.295*(((2*pi)/M3)-((fpass*2*pi)+(ftrans(i)*2*pi))-(fpass*2*pi))))+1;
c3=((N_tr-N_be-N_ma)/N_tr)*100;
 
%N_tr=((A-8)/(2.285*ftrans(i)*2*pi))+1;
N_be=((A-8)/(2.285*M4*ftrans(i)*2*pi))+1;
N_ma=((A-8)/(2.295*(((2*pi)/M4)-((fpass*2*pi)+(ftrans(i)*2*pi))-(fpass*2*pi))))+1;
c4=((N_tr-N_be-N_ma)/N_tr)*100;
    
    if(max([c2,c3,c4])==c2)
      optimum_val(i)=2;
    elseif(max([c2,c3,c4])==c3)
        optimum_val(i)=3;
    else
        optimum_val(i)=4;
    end
    
    if(((2*pi)-(M2*ftrans(i)*2*pi)-(2*M2*fpass*2*pi))<=0)
    optimum_val(i)=optimum_val(i-1);
    elseif(((2*pi)-(M3*ftrans(i)*2*pi)-(2*M3*fpass*2*pi))<=0)
    optimum_val(i)=optimum_val(i-1); 
    elseif(((2*pi)-(M4*ftrans(i)*2*pi)-(2*M4*fpass*2*pi))<=0)
    optimum_val(i)=optimum_val(i-1);    
    end
    
end  
figure   
plot(ftrans,optimum_val)
xlabel('Transition Region Bandwidth(X 2Xpi)')
ylabel('Optimum Expansion Factor(M)')
axis([0 0.1 0 4])
grid on

%Code illustrating the high pass filter design-
close all
clear all
expansion_factor=3;
   wc =0.2*pi*expansion_factor; wc2 = pi;
   alp = 25;
   n = 0:1:ceil(2*alp);
   nfine= linspace(0,ceil(2*alp),300);
   wc1 = wc; wc2 = pi;
   hlow = (wc/pi)*sinc((wc/pi)*(n-alp)); %% Using Matlab function sinc(x) = sin(pi*x)/(pi*x)
   hlowenv = (wc/pi)*sinc((wc/pi)*(nfine-alp));
   h  = sinc(n-alp) - hlow; 
   henv = sinc(nfine-alp) - hlowenv;
   %
   z = roots(h);
   a=[1 zeros(1,length(h)-1)];
   p = roots(a);
  %interpolate the high pass filter
 
h_interpolated=upsample(h,expansion_factor);
for r = 1:expansion_factor-1
   h_interpolated(length(h_interpolated))=[];
end
n_interpolated=0:1:length(h_interpolated)-1;
b=[1 zeros(1,length(h_interpolated)-1)];
 
  %Print out zero locations
  disp(sprintf('Locations of zeros in z-plane for alpha = %4.2f',alp))
  for k=1:length(z)
      disp(sprintf('r=%7.4f (1/r=%7.4f) theta=%7.4f', abs(z(k)), 1/abs(z(k)), angle(z(k)) ));
  end
  
    %
   N=512;
   w=linspace(0,pi,N);
   H_be=freqz(h_interpolated,b,w);
   H=freqz(h,a,w);
       wc1_ma =0.5*pi;
   alp_ma = 25;
   n_ma = 0:1:ceil(2*alp_ma);
   nfine_ma= linspace(0,ceil(2*alp_ma),300);
   %
   % For high-pass case, subtract a low-pass from an all-pass
   wc2 = pi;
   hlow_ma = (wc1_ma/pi)*sinc((wc1_ma/pi)*(n_ma-alp_ma)); %% Using Matlab function sinc(x) = sin(pi*x)/(pi*x)
   hlowenv_ma = (wc1_ma/pi)*sinc((wc1_ma/pi)*(nfine_ma-alp_ma));
   
   h_ma  = sinc(n_ma-alp_ma) - hlow_ma; 
   henv_ma = sinc(nfine-alp_ma) - hlowenv_ma;
   %
   c=[1 zeros(1,length(h)-1)];
   H_ma=freqz(h_ma,c,w);
   
   y=abs(H_ma).*abs(H_be);
   
   figure
   subplot(411), plot(w/pi,abs(H))
   xlabel('w'),ylabel('|H(w)|')
   title(sprintf('Frequency plot for prototype filter, alpha = %4.2f, wc1 = %4.2f pi, wc2 = %4.2f pi',alp,            wc1/pi,wc2/pi));
   subplot(412), plot(w/pi,abs(H_be))
   xlabel('w'),ylabel('|H(w)|')
   title('Frequency plot for bandedge shaping filter');
   subplot(413), plot(w/pi,abs(H_ma))
   xlabel('w'),ylabel('|H(w)|')
   title('Frequency plot for masking filter');
   subplot(414), plot(w/pi,abs(y))
   xlabel('w'),ylabel('|H(w)|')
   title('Frequency plot for resulting filter');
   figure
   subplot(311), stem(n,h),
   hold on
   plot(nfine,henv), hold off
   xlabel('n'), ylabel('h(n)'), title('Imp resp')
   title(sprintf('Impulse response and zero locations for alpha = %4.2f and wc = %4.2f pi, wc2 = %4.2f pi',alp,    wc1/pi,wc2/pi));
   subplot(312), zplane(z,p), %% title('Zero locations for H(z), alpha = ')
   subplot(313),stem(n_interpolated,h_interpolated)
   
   
 

%Code illustrating a practical FIR application-
fp=0.1; %0.1 hertz
fs=1; %1 hertz
t=0:1/fs:200;
x = cos(0.2*t);
x_ns=x'+0.25*randn(length(t),1); % noisy waveform
subplot(211),plot(t,x_ns)
xlabel('n')
ylabel('Noisy signal')
y_freq=fft(x_ns);
w=linspace(0,pi,201);
%subplot(512),plot(w/pi,abs(y_freq))
 
D1 = mean(grpdelay(fir1(61,0.6,'low',kaiser(62,5.65326))));
D=ceil(D1);
[b1 a1]=fir1(61,0.6,'low',kaiser(62,5.65326));
y1 = filter(b1,a1,[x_ns; zeros(D,1)]); 
y1=y1(D+1:end);
 
D2 = mean(grpdelay(fir1(32,0.2,'low',kaiser(33,5.65326))));
D=ceil(D2);
[b2 a2]=fir1(32,0.1,'low',kaiser(33,5.65326));
y2=filter(b2,a2,[y1; zeros(D,1)]);
y2=y2(D+1:end);
subplot(212),plot(t,y2);
xlabel('n')
ylabel('Filtered Signal')
