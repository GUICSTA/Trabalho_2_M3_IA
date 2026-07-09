import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Main {

    // Estruturas para o pré-processamento
    static double[] minValues = new double[17];
    static double[] maxValues = new double[17];
    static Map<Integer, List<String>> categories = new HashMap<>();

    // Índices das colunas no ficheiro CSV baseados na documentação
    static int[] numericCols = {0, 5, 9, 11, 12, 13, 14};
    static int[] binaryCols = {4, 6, 7, 16};
    static int[] categoricalCols = {1, 2, 3, 8, 10, 15};

    public static void main(String[] args) {

        try {

            System.out.println("-> A mapear os dados para pré-processamento...");
            analisarDados("bank-full.csv");


            System.out.println("-> A processar bank.csv (Treino - 10%)...");
            List<double[]> trainData = processarCSV("bank.csv", true);

            System.out.println("-> A processar bank-full.csv (Teste - Completo)...");
            List<double[]> testData = processarCSV("bank-full.csv", false);

            int numEntradas = trainData.get(0).length - 1;
            System.out.println("-> Total de entradas na rede após One-Hot Encoding: " + numEntradas + " atributos.");


            System.out.println("\n--- A TREINAR ABORDAGEM 1 (1 Neurónio Binário) ---");
            Neuronio nBinario = new Neuronio(numEntradas);

            int epocas = 100;
            for(int e = 0; e < epocas; e++) {
                Collections.shuffle(trainData);
                for (double[] linha : trainData) {
                    double[] in = Arrays.copyOfRange(linha, 0, numEntradas);
                    double target = linha[numEntradas];

                    double output = nBinario.executa(in);
                    double erro = target - output;

                    if(erro != 0) {
                        nBinario.calculaDelta(erro, in);
                        for(int i = 0; i < nBinario.w.length; i++){
                            nBinario.w[i] += nBinario.dw[i];
                        }
                        nBinario.zeraDW();
                    }
                }
            }

            // Abordagem 1
            int vp1 = 0, vn1 = 0, fp1 = 0, fn1 = 0;
            for (double[] linha : testData) {
                double[] in = Arrays.copyOfRange(linha, 0, numEntradas);
                double target = linha[numEntradas];
                double output = nBinario.executa(in);

                if(target == 1 && output == 1) vp1++;
                else if(target == 0 && output == 0) vn1++;
                else if(target == 0 && output == 1) fp1++;
                else if(target == 1 && output == 0) fn1++;
            }

            System.out.println("Resultados (1 Neurónio):");
            System.out.println(" Verdadeiros Positivos (Clientes previstos como investidores que o fizeram): " + vp1);
            System.out.println(" Falsos Positivos: " + fp1);
            System.out.println(" Precisão Global: " + String.format("%.2f", (double)(vp1+vn1)/testData.size() * 100) + "%");

            System.out.println("\n--- A TREINAR ABORDAGEM 2 (2 Neurónios) ---");
            Neuronio nYes = new Neuronio(numEntradas); // Treinado para reconhecer investidores
            Neuronio nNo = new Neuronio(numEntradas);  // Treinado para reconhecer não-investidores

            for(int e = 0; e < epocas; e++) {
                Collections.shuffle(trainData);
                for (double[] linha : trainData) {
                    double[] in = Arrays.copyOfRange(linha, 0, numEntradas);
                    double target = linha[numEntradas];

                    double targetYes = target == 1 ? 1 : 0;
                    double targetNo = target == 0 ? 1 : 0;

                    double outYes = nYes.executa(in);
                    double outNo = nNo.executa(in);

                    if(targetYes - outYes != 0) {
                        nYes.calculaDelta(targetYes - outYes, in);
                        for(int i = 0; i < nYes.w.length; i++) nYes.w[i] += nYes.dw[i];
                        nYes.zeraDW();
                    }
                    if(targetNo - outNo != 0) {
                        nNo.calculaDelta(targetNo - outNo, in);
                        for(int i = 0; i < nNo.w.length; i++) nNo.w[i] += nNo.dw[i];
                        nNo.zeraDW();
                    }
                }
            }

            int vp2 = 0, vn2 = 0, fp2 = 0, fn2 = 0;
            for (double[] linha : testData) {
                double[] in = Arrays.copyOfRange(linha, 0, numEntradas);
                double target = linha[numEntradas];

                double somaYes = 0, somaNo = 0;
                for(int i = 0; i < in.length; i++) {
                    somaYes += in[i] * nYes.w[i];
                    somaNo += in[i] * nNo.w[i];
                }
                somaYes += 1 * nYes.w[numEntradas];
                somaNo += 1 * nNo.w[numEntradas];

                double predictedTarget = (somaYes > somaNo) ? 1 : 0;

                if(target == 1 && predictedTarget == 1) vp2++;
                else if(target == 0 && predictedTarget == 0) vn2++;
                else if(target == 0 && predictedTarget == 1) fp2++;
                else if(target == 1 && predictedTarget == 0) fn2++;
            }

            System.out.println("Resultados (2 Neurónios - Pseudo Softmax):");
            System.out.println("Verdadeiros Positivos (Clientes previstos como investidores que o fizeram): " + vp2);
            System.out.println("Falsos Positivos: " + fp2);
            System.out.println("Precisão Global: " + String.format("%.2f", (double)(vp2+vn2)/testData.size() * 100) + "%");

        } catch (Exception e) {
            System.out.println("Erro ao ler os ficheiros. Verifique se o bank.csv e bank-full.csv estão na mesma pasta.");
            e.printStackTrace();
        }
    }

    static void analisarDados(String file) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line = br.readLine();

        Arrays.fill(minValues, Double.MAX_VALUE);
        Arrays.fill(maxValues, Double.MIN_VALUE);
        for(int c : categoricalCols) categories.put(c, new ArrayList<>());

        while ((line = br.readLine()) != null) {
            String[] val = line.replace("\"", "").split(";");

            for(int i : numericCols) {
                double v = Double.parseDouble(val[i]);
                if(v < minValues[i]) minValues[i] = v;
                if(v > maxValues[i]) maxValues[i] = v;
            }

            for(int i : categoricalCols) {
                if(!categories.get(i).contains(val[i])) {
                    categories.get(i).add(val[i]);
                }
            }
        }
        br.close();
    }

    static List<double[]> processarCSV(String file, boolean oversampleYes) throws IOException {
        List<double[]> data = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line = br.readLine();

        int totalCols = numericCols.length + binaryCols.length - 1;
        for(int c : categoricalCols) totalCols += categories.get(c).size();

        while ((line = br.readLine()) != null) {
            String[] val = line.replace("\"", "").split(";");
            double[] processed = new double[totalCols + 1]; // +1 para guardar o target no fim do vetor
            int index = 0;

            for(int i : numericCols) {
                double v = Double.parseDouble(val[i]);
                // (valor - min) / (max - min)
                processed[index++] = (v - minValues[i]) / (maxValues[i] - minValues[i]);
            }

            // yes = 1, no = 0
            for(int i : binaryCols) {
                if(i != 16) {
                    processed[index++] = val[i].equals("yes") ? 1.0 : 0.0;
                }
            }

            for(int i : categoricalCols) {
                List<String> cats = categories.get(i);
                for(String c : cats) {
                    processed[index++] = val[i].equals(c) ? 1.0 : 0.0;
                }
            }

            double target = val[16].equals("yes") ? 1.0 : 0.0;
            processed[index] = target;

            data.add(processed);

            if(oversampleYes && target == 1.0) {
                data.add(processed);
                data.add(processed);
                data.add(processed);
            }
        }
        br.close();
        return data;
    }
}