<<<<<<< HEAD
package com.itheima.d3_transform_stream;


import java.io.*;

/**
 * 目标：掌握字符输入转换流的作用。
 */
public class InputStreamReaderTest2 {
    public static void main(String[] args) {
        try (
                // 1、得到文件的原始字节流（GBK的字节流形式）
                InputStream is = new FileInputStream("io-app2/src/itheima06.txt");
                // 2、把原始的字节输入流按照指定的字符集编码转换成字符输入流
                Reader isr = new InputStreamReader(is, "GBK");
                // 3、把字符输入流包装成缓冲字符输入流
                BufferedReader br = new BufferedReader(isr);
                ){
            String line;
            while ((line = br.readLine()) != null){
                System.out.println(line);
            }


        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
=======
package com.itheima.d3_transform_stream;


import java.io.*;

/**
 * 目标：掌握字符输入转换流的作用。
 */
public class InputStreamReaderTest2 {
    public static void main(String[] args) {
        try (
                // 1、得到文件的原始字节流（GBK的字节流形式）
                InputStream is = new FileInputStream("io-app2/src/itheima06.txt");
                // 2、把原始的字节输入流按照指定的字符集编码转换成字符输入流
                Reader isr = new InputStreamReader(is, "GBK");
                // 3、把字符输入流包装成缓冲字符输入流
                BufferedReader br = new BufferedReader(isr);
                ){
            String line;
            while ((line = br.readLine()) != null){
                System.out.println(line);
            }


        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
