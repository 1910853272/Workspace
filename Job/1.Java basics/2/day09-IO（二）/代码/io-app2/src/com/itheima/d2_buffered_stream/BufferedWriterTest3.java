<<<<<<< HEAD
package com.itheima.d2_buffered_stream;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.Writer;

/**
 * 目标：掌握字符缓冲输出流的用法。
 */
public class BufferedWriterTest3 {
    public static void main(String[] args) {
        try (
                Writer fw = new FileWriter("io-app2/src/itheima05out.txt", true);
                // 创建一个字符缓冲输出流管道包装原始的字符输出流
                BufferedWriter bw = new BufferedWriter(fw);
        ){

            bw.write('a');
            bw.write(97);
            bw.write('磊');
            bw.newLine();

            bw.write("我爱你中国abc");
            bw.newLine();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
=======
package com.itheima.d2_buffered_stream;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.Writer;

/**
 * 目标：掌握字符缓冲输出流的用法。
 */
public class BufferedWriterTest3 {
    public static void main(String[] args) {
        try (
                Writer fw = new FileWriter("io-app2/src/itheima05out.txt", true);
                // 创建一个字符缓冲输出流管道包装原始的字符输出流
                BufferedWriter bw = new BufferedWriter(fw);
        ){

            bw.write('a');
            bw.write(97);
            bw.write('磊');
            bw.newLine();

            bw.write("我爱你中国abc");
            bw.newLine();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
