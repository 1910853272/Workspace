function eqs = DAE_pre(t,in2,in3,param3)
%DAE_PRE
%    EQS = DAE_PRE(T,IN2,IN3,PARAM3)

%    This function was generated by the Symbolic Math Toolbox version 8.4.
%    15-Jun-2022 12:10:10

YP88 = in3(10,:);
YP89 = in3(11,:);
YP90 = in3(12,:);
YP91 = in3(13,:);
YP92 = in3(14,:);
YP93 = in3(15,:);
YP94 = in3(16,:);
YP95 = in3(17,:);
YP96 = in3(18,:);
ji1 = in2(39,:);
jm1 = in2(19,:);
jm2 = in2(20,:);
jm3 = in2(21,:);
jm4 = in2(22,:);
jm5 = in2(23,:);
jm6 = in2(24,:);
jm7 = in2(25,:);
jm8 = in2(26,:);
jm9 = in2(27,:);
jm10 = in2(28,:);
jm11 = in2(29,:);
jm12 = in2(30,:);
jm13 = in2(31,:);
jm14 = in2(32,:);
jm15 = in2(33,:);
jm16 = in2(34,:);
jm17 = in2(35,:);
jm18 = in2(36,:);
jm19 = in2(37,:);
jm20 = in2(38,:);
phin2 = in2(10,:);
phin3 = in2(11,:);
phin4 = in2(12,:);
phin5 = in2(13,:);
phin6 = in2(14,:);
phin7 = in2(15,:);
phin8 = in2(16,:);
phin9 = in2(17,:);
phin10 = in2(18,:);
vn2 = in2(1,:);
vn3 = in2(2,:);
vn4 = in2(3,:);
vn5 = in2(4,:);
vn6 = in2(5,:);
vn7 = in2(6,:);
vn8 = in2(7,:);
vn9 = in2(8,:);
vn10 = in2(9,:);
t2 = phin2.*1.343087320339522e+8;
t3 = phin3.*1.099241278683989e+8;
t4 = phin6.*1.099241278683989e+8;
t5 = phin3.*1.080011125263798e+8;
t6 = phin4.*1.080011125263798e+8;
t10 = phin7.*1.066430596974853e+8;
t11 = phin8.*1.066430596974853e+8;
t12 = phin3.*2.008200317635152e+8;
t13 = phin5.*2.008200317635152e+8;
t14 = phin8.*1.006167690638146e+8;
t15 = phin9.*1.006167690638146e+8;
t16 = phin10.*1.903251492417614e+8;
t17 = phin4.*5.494271512186822e+7;
t18 = phin8.*5.494271512186822e+7;
t19 = phin2.*1.526667475238422e+8;
t20 = phin7.*1.526667475238422e+8;
t21 = phin2.*3.564607648439303e+7;
t22 = phin3.*3.564607648439303e+7;
t23 = phin6.*3.60926715381228e+7;
t24 = phin7.*3.60926715381228e+7;
t25 = phin8.*9.966080888995475e+7;
t26 = phin5.*8.946343712986059e+7;
t27 = phin6.*8.946343712986059e+7;
t31 = phin6.*9.384732577485518e+7;
t32 = phin4.*8.186000213550435e+7;
t33 = phin7.*8.186000213550435e+7;
t34 = phin9.*3.706795745997334e+7;
t35 = phin10.*3.706795745997334e+7;
t36 = phin2.*3.277743519504064e+7;
t37 = phin8.*3.277743519504064e+7;
t38 = phin9.*1.139937803121177e+8;
t39 = phin5.*9.612722876446714e+7;
t40 = phin7.*9.612722876446714e+7;
t46 = phin4.*1.24542785588072e+8;
t47 = phin5.*1.24542785588072e+8;
t7 = -t2;
t8 = -t4;
t9 = -t6;
t28 = -t11;
t29 = -t12;
t30 = -t15;
t41 = -t18;
t42 = -t20;
t43 = -t22;
t44 = -t24;
t45 = -t25;
t48 = -t27;
t49 = -t31;
t50 = -t32;
t51 = -t35;
t52 = -t37;
t53 = -t38;
t54 = -t40;
t55 = -t47;
t57 = t16+5.544917849540196e+7;
t56 = t7+2.768205698061579e+7;
t58 = t45+1.184920868392741e+7;
t59 = t49+3.338104044315648e+7;
t60 = t53+2.734440753155296e+7;
t62 = 1.0./sqrt(t57);
t66 = t5+t9+3.934401910901023e+7;
t67 = t3+t8+3.260693120281367e+7;
t68 = t13+t29+5.04031289252896e+7;
t69 = t19+t42+2.68485921376566e+7;
t70 = t10+t28+3.828592845851474e+7;
t72 = t17+t41+5.235788326548361e+6;
t73 = t14+t30+1.376478127405948e+7;
t75 = t39+t54+3.096304376159295e+7;
t76 = t23+t44+1.701140811211304e+6;
t77 = t21+t43+1.97347851675335e+6;
t78 = t46+t55+2.98112982551868e+7;
t79 = t33+t50+7.835641321831161e+6;
t82 = t36+t52+2.616731367712787e+6;
t83 = t26+t48+1.602196255173322e+7;
t86 = t34+t51+1.938526589383156e+6;
t61 = 1.0./sqrt(t56);
t63 = 1.0./sqrt(t59);
t64 = 1.0./sqrt(t60);
t65 = 1.0./sqrt(t58);
t71 = 1.0./sqrt(t67);
t74 = 1.0./sqrt(t66);
t87 = 1.0./sqrt(t68);
t88 = 1.0./sqrt(t69);
t90 = 1.0./sqrt(t70);
t94 = 1.0./sqrt(t75);
t95 = 1.0./sqrt(t76);
t96 = 1.0./sqrt(t77);
t97 = 1.0./sqrt(t72);
t98 = 1.0./sqrt(t73);
t103 = 1.0./sqrt(t78);
t105 = 1.0./sqrt(t83);
t108 = 1.0./sqrt(t86);
t111 = 1.0./sqrt(t79);
t118 = 1.0./sqrt(t82);
t80 = YP89.*t74;
t81 = YP90.*t74;
t84 = YP89.*t71;
t85 = YP92.*t71;
t92 = YP89.*t87;
t93 = YP91.*t87;
t99 = YP88.*t88;
t100 = YP93.*t88;
t101 = YP93.*t90;
t102 = YP94.*t90;
t106 = YP90.*t97;
t107 = YP94.*t97;
t109 = YP94.*t98;
t110 = YP95.*t98;
t112 = YP91.*t94;
t113 = YP93.*t94;
t114 = YP92.*t95;
t115 = YP93.*t95;
t116 = YP88.*t96;
t117 = YP89.*t96;
t121 = YP90.*t103;
t122 = YP91.*t103;
t123 = YP95.*t108;
t124 = YP96.*t108;
t127 = YP90.*t111;
t128 = YP93.*t111;
t129 = YP88.*t118;
t130 = YP94.*t118;
t134 = YP91.*t105;
t135 = YP92.*t105;
t89 = -t80;
t91 = -t84;
t104 = -t93;
t119 = -t99;
t120 = -t101;
t125 = -t106;
t126 = -t109;
t131 = -t112;
t132 = -t114;
t133 = -t116;
t136 = -t121;
t137 = -t123;
t138 = -t128;
t139 = -t129;
t140 = -t134;
eqs = [ji1-t100-t117-t130+YP88.*(t61+t88+t96+t118);-t81-t85+t104+t133+YP89.*(t71+t74+t87+t96);t89-t107-t122+t138+YP90.*(t74+t97+t103+t111);-t92-t113-t135+t136+YP91.*(t87+t94+t103+t105);t91-t115+t140+YP92.*(t63+t71+t95+t105);-t102+t119-t127+t131+t132+YP93.*(t88+t90+t94+t95+t111);-t110+t120+t125+t139+YP94.*(t65+t90+t97+t98+t118);-t124+t126+YP95.*(t64+t98+t108);t137+YP96.*(t62+t108);YP88-vn2;YP89-vn3;YP90-vn4;YP91-vn5;YP92-vn6;YP93-vn7;YP94-vn8;YP95-vn9;YP96-vn10;-jm1+YP88.*t61;-jm2+t117+t133;-jm3+t81+t89;-jm4+t92+t104;-jm5+t122+t136;-jm6+YP92.*t63;-jm7+t85+t91;-jm8+t135+t140;-jm9+t100+t119;-jm10+t127+t138;-jm11+t113+t131;-jm12+t115+t132;-jm13+YP94.*t65;-jm14+t130+t139;-jm15+t107+t125;-jm16+t102+t120;-jm17+YP95.*t64;-jm18+t110+t126;-jm19-YP96.*t62;-jm20+t124+t137;-param3+vn2];
