clear;clc;
wp=[0.5 0.6];N=20;
h=fir1(N,wp,'bandpass');fs=6700;
[H,f]=freqz(h,1,512,fs);
f1=100;f2=800;f3=1800;ts=1/fs;n=0:200;
x=sin(2*pi*f1*n*ts)+sin(2*pi*f2*n*ts)+0.5*sin(2*pi*f3*n*ts);
y=filter(h,1,x);
figure;
subplot(2,1,1);
plot(f,20*log10(abs(H)));
subplot(2,1,2);
plot(f,180/pi*unwrap(angle(H)));
figure;
subplot(2,1,1);
plot(n*ts,x);
subplot(2,1,2);
plot(n*ts,y);
save('kernal_bandpass_high.mat','h');