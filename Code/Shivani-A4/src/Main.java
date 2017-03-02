import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.StringTokenizer;

/**
 * 
 * This class implements Decision Tree (C4.5) as basic method, Random Forest as the ensemble version of the classification method.
 * @author Shivani Sharma
 *
 */
public class Main {
	
	static String TRAINING_DATASET_LOCATION = "C://Users/Lucky/Downloads/Assignment 4/A4. Package/Datasets/breast_cancer, training.txt";
	static String TEST_DATASET_LOCATION = "C://Users/Lucky/Downloads/Assignment 4/A4. Package/Datasets/breast_cancer, test.txt";
	static String OUTOUT_LOCATION = "C://Users/Lucky/Downloads/Assignment 4/A4. Package/Output/";
	static String OUTPUT_BASIC_FILE_NAME = "Basic - breast cancer.txt";
	static String OUTPUT_ENSEMBLE_FILE_NAME = "Ensemble - breast cancer.txt";
	static int NUMBER_OF_ATTRIBUTES_IN_DATASET = 10;
	
	/**
	 * Constants for Random Forest method
	 */
	static int K = 10000; //Number of training set samples and decision trees that needs to be made
	static int F = 2; //Number of random attributes that needs to be used to determine the split at each node
	
	
	static Set<Integer> initialAttributeSet = new HashSet<Integer>();
	
	
	public static void main(String[] args){
		//Read in a training dataset and test dataset and store it in memory
		
		Map<String, List<Tuple>> trainingTuplesMap = new HashMap<String, List<Tuple>>();
		
		//This list will be used in Random Forest code
		List<Tuple> trainingTuplesList = new ArrayList<Tuple>();
		
		List<Tuple> testTuples = new ArrayList<Tuple>();
		Set<Integer> attributeSet = new HashSet<Integer>();
		BufferedReader bufferedReader = null;
		try{
			File file = new File(TRAINING_DATASET_LOCATION);
			bufferedReader = new BufferedReader(new FileReader(file));	
			while (bufferedReader.ready()) {    		
	    		String line=bufferedReader.readLine();
	    		if(line == null || line.equals("")){
	    			continue;
	    		}
	    		Tuple aTuple = new Tuple(line);
	    		Set<Integer> attributeSetForATuple = aTuple.getAttributeSet();
	    		attributeSet.addAll(attributeSetForATuple);
	    		
	    		trainingTuplesList.add(aTuple);
	    		
	    		//Add into the training tuple map which has key as the label
	    		if(trainingTuplesMap.get(aTuple.getLabel()) == null){
	    			trainingTuplesMap.put(aTuple.getLabel(), new ArrayList<Tuple>());
	    			trainingTuplesMap.get(aTuple.getLabel()).add(aTuple);
	    		} else {
	    			trainingTuplesMap.get(aTuple.getLabel()).add(aTuple);
	    		}
	    	}	    	
		} catch(Exception e){
			throw new RuntimeException("Exception Occured in reading training file.", e);
		} finally {
			try {
				bufferedReader.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		//Make a copy of attribute set as the generate decision tree method remove attributes from the set and we 
		// have to use the attribute set for random forest
		Set<Integer> originalAttributeSet = new HashSet<Integer>(attributeSet);
			
		// Generate the decision tree using basic algorithm
		Node N = generateDecisionTree(trainingTuplesMap, attributeSet, false);
//		System.out.println(N);
		
		// Read the test dataset and its labels
		StringBuffer outputContent = new StringBuffer();
		double truePredections = 0;
		double falsePredections = 0;
		double totalInstances = 0;
		double TPCount = 0;
		double TNCount = 0;
		double FPCount = 0;
		double FNCount = 0;
		
		double PCount = 0;
		double NCount = 0;
		
		StringBuffer finalOutput = new StringBuffer();
		try{
			File file = new File(TEST_DATASET_LOCATION);
			bufferedReader = new BufferedReader(new FileReader(file));	
			Tuple tuple = null;
			String predictedLabel = null;
			boolean isLabelPredictionCorrect;
			while (bufferedReader.ready()) {    		
	    		String line=bufferedReader.readLine();
	    		if(line == null || line.equals("")){
	    			continue;
	    		}
	    		tuple = new Tuple(line);
	    		testTuples.add(tuple);
	    		outputContent.append(line).append(" -> ");
	    		try{
	    			if(tuple.getLabel().equals("+1")){
	    				PCount++;
	    			} else {
	    				NCount++;
	    			}
	    			predictedLabel = predictClassForTuple(N, tuple);
	    			isLabelPredictionCorrect = predictedLabel.equals(tuple.getLabel());
	    			if(isLabelPredictionCorrect){
	    				truePredections++;
	    				if(predictedLabel.equals("+1")){
	    					TPCount++;
	    				} else {
	    					TNCount++;
	    				}
	    			} else {
	    				falsePredections++;
	    				if(predictedLabel.equals("+1")){
	    					FPCount++;
	    				} else {
	    					FNCount++;
	    				}
	    			}
	    		} catch(Exception e){
//	    			System.out.println("!!! Error - Exception occured. "+e.getMessage());
	    			isLabelPredictionCorrect = false;
	    			falsePredections++;
	    		}
	    		
	    		
	    		outputContent.append(isLabelPredictionCorrect);
	    		outputContent.append(System.getProperty("line.separator"));
	    		
	    	}	
			totalInstances = testTuples.size();
			
			double accuracy = (truePredections *100)/totalInstances;
			double errorRate = falsePredections / totalInstances;
			double sensitivity = TPCount/(PCount);
			double speficity = TNCount / (NCount);
			double recall = TPCount / (TPCount+FNCount);
			double precision = TPCount / (TPCount + FPCount);
			
			
			double F1 = (2*precision*recall) / (precision + recall);
			double FB0Half = ((1+ (0.5 * 0.5 ))* precision * recall) / (((0.5 * 0.5 )*precision) + recall);
			
			double FB2 = ((1+ (2 * 2 ))* precision * recall) / (((2 * 2 )*precision) + recall);
			
			finalOutput.append("Total instances: ").append(totalInstances).append(System.getProperty("line.separator"));
			finalOutput.append("True predictions: ").append(truePredections).append(System.getProperty("line.separator"));
			finalOutput.append("False predictions: ").append(falsePredections).append(System.getProperty("line.separator"));
			
			finalOutput.append("PCount: ").append(PCount).append(System.getProperty("line.separator"));
			finalOutput.append("NCount: ").append(NCount).append(System.getProperty("line.separator"));
			
			finalOutput.append("TPCount: ").append(TPCount).append(System.getProperty("line.separator"));
			finalOutput.append("TNCount").append(TNCount).append(System.getProperty("line.separator"));
			finalOutput.append("FPCount").append(FPCount).append(System.getProperty("line.separator"));
			finalOutput.append("FNCount").append(FNCount).append(System.getProperty("line.separator"));
			
			finalOutput.append("Accuracy: ").append(accuracy).append("%").append(System.getProperty("line.separator"));
			finalOutput.append("Error Rate: ").append(errorRate).append(System.getProperty("line.separator"));
			
			finalOutput.append("sensitivity: ").append(sensitivity).append(System.getProperty("line.separator"));
			finalOutput.append("speficity: ").append(speficity).append(System.getProperty("line.separator"));
			finalOutput.append("precision: ").append(precision).append(System.getProperty("line.separator"));
			finalOutput.append("recall: ").append(recall).append(System.getProperty("line.separator"));
			finalOutput.append("F1: ").append(F1).append(System.getProperty("line.separator"));
			finalOutput.append("FB0Half: ").append(FB0Half).append(System.getProperty("line.separator"));
			finalOutput.append("FB2: ").append(FB2).append(System.getProperty("line.separator")).append(System.getProperty("line.separator"));
			
			
			finalOutput.append("***********************************").append(System.getProperty("line.separator"));
			finalOutput.append(outputContent);
			
			System.out.println(finalOutput);
			
			writeIntoFile(Main.OUTPUT_BASIC_FILE_NAME, finalOutput.toString());
			
			
			
			
		} catch(Exception e){
			throw new RuntimeException("Exception Occured in reading test file.", e);
		}  finally {
			try {
				bufferedReader.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		/**
		 * Random Forest method as the ensembel version of the classification method starts here
		 */
		// Create k number of samples from training Set D of same size by sampling with replacement
		Map[] sampleTrainingTupleSets = new Map[Main.K + 1];
		Node[] decisionTrees = new Node[Main.K + 1];
		Random rand = new Random();
		
		for(int i = 0 ; i < Main.K; i++){
			Map<String, List<Tuple>> Di = new HashMap<String, List<Tuple>>();
			for(int j = 0; j < trainingTuplesList.size(); j++){
				// nextInt excludes the top value so we should pass the training tuple list size.
				// if size is 100 then the nextInt method can return any value from 0 to 99 (including 0 and  99)
				Tuple t = trainingTuplesList.get(rand.nextInt(trainingTuplesList.size()));
				//Add into the training tuple map which has key as the label
	    		if(Di.get(t.getLabel()) == null){
	    			Di.put(t.getLabel(), new ArrayList<Tuple>());
	    			Di.get(t.getLabel()).add(t);
	    		} else {
	    			Di.get(t.getLabel()).add(t);
	    		}
			}
			sampleTrainingTupleSets[i] = Di;
			Set attributeList = new HashSet(originalAttributeSet);
			System.out.println("Generating deciosion tree for Di. i = "+i);
			decisionTrees[i] = generateDecisionTree(Di, attributeList, true);
		}
		System.out.println("Created decisionTrees");
		System.out.println("Now reading test data");
		// Read the test dataset and its labels
		outputContent = new StringBuffer();
		truePredections = 0;
		falsePredections = 0;
		totalInstances = 0;
		
		TPCount = 0;
		TNCount = 0;
		FPCount = 0;
		FNCount = 0;
		PCount = 0;
		NCount = 0;
		
		testTuples = new ArrayList<Tuple>();
		
		finalOutput = new StringBuffer();
		try{
			File file = new File(TEST_DATASET_LOCATION);
			bufferedReader = new BufferedReader(new FileReader(file));	
			Tuple tuple = null;
			String predictedLabel = null;
			boolean isLabelPredictionCorrect;
			while (bufferedReader.ready()) {    		
	    		String line=bufferedReader.readLine();
	    		if(line == null || line.equals("")){
	    			continue;
	    		}
	    		tuple = new Tuple(line);
	    		if(tuple.getLabel().equals("+1")){
    				PCount++;
    			} else {
    				NCount++;
    			}
	    		
	    		testTuples.add(tuple);
	    		outputContent.append(line).append(" -> ");
	    		try{
	    			//Maintains vote count for each class
	    			Map<String, Integer> voteMap = new HashMap<String, Integer>();
	    			for(Node aDecisionTree : decisionTrees){
	    				try{
	    					
	    					predictedLabel = predictClassForTuple(aDecisionTree, tuple);
	    					if(voteMap.get(predictedLabel) == null){
		    					voteMap.put(predictedLabel, 1);
		    				} else {
		    					voteMap.put(predictedLabel, voteMap.get(predictedLabel)+1);
		    				}
	    				} catch(Exception e){
	    					//TODO: Handle exception
	    					
	    				}
	    			}
	    			String majorityClass = null;
	    			int majorityClassVoteCount = 0;
	    			for(String C : voteMap.keySet()){
	    				if(majorityClass == null){
	    					majorityClass = C;
	    					majorityClassVoteCount = voteMap.get(C);
	    				} else if(majorityClassVoteCount < voteMap.get(C)){
	    					majorityClass = C;
	    					majorityClassVoteCount = voteMap.get(C);
	    				}
	    			}
	    			predictedLabel = majorityClass;
	    			
	    			isLabelPredictionCorrect = predictedLabel.equals(tuple.getLabel());
	    			if(isLabelPredictionCorrect){
	    				truePredections++;
	    				if(predictedLabel.equals("+1")){
	    					TPCount++;
	    				} else {
	    					TNCount++;
	    				}
	    			} else {
	    				falsePredections++;
	    				if(predictedLabel.equals("+1")){
	    					FPCount++;
	    				} else {
	    					FNCount++;
	    				}
	    			}
	    		} catch(Exception e){
//	    			System.out.println("!!! Error - Exception occured. "+e.getMessage());
	    			isLabelPredictionCorrect = false;
	    			falsePredections++;
	    		}
	    		
	    		
	    		outputContent.append(isLabelPredictionCorrect);
	    		outputContent.append(System.getProperty("line.separator"));
	    		
	    	}	
			totalInstances = testTuples.size();
			
			double accuracy = (truePredections *100)/totalInstances;
			double errorRate = falsePredections / totalInstances;
			double sensitivity = TPCount/(PCount);
			double speficity = TNCount / (NCount);
			double precision = TPCount / (TPCount + FPCount);
			double recall = TPCount / (TPCount+FNCount);
			
			double F1 = (2*precision*recall) / (precision + recall);
			double FB0Half = ((1+ (0.5 * 0.5 ))* precision * recall) / (((0.5 * 0.5 )*precision) + recall);
			
			double FB2 = ((1+ (2 * 2 ))* precision * recall) / (((2 * 2 )*precision) + recall);
			
			
			finalOutput.append("Total instances: ").append(totalInstances).append(System.getProperty("line.separator"));
			finalOutput.append("True predictions: ").append(truePredections).append(System.getProperty("line.separator"));
			finalOutput.append("False predictions: ").append(falsePredections).append(System.getProperty("line.separator"));
			
			finalOutput.append("PCount: ").append(PCount).append(System.getProperty("line.separator"));
			finalOutput.append("NCount: ").append(NCount).append(System.getProperty("line.separator"));
			
			finalOutput.append("Accuracy: ").append(accuracy).append("%").append(System.getProperty("line.separator"));
			finalOutput.append("Error Rate: ").append(errorRate).append(System.getProperty("line.separator"));
			
			finalOutput.append("sensitivity: ").append(sensitivity).append(System.getProperty("line.separator"));
			finalOutput.append("speficity: ").append(speficity).append(System.getProperty("line.separator"));
			finalOutput.append("precision: ").append(precision).append(System.getProperty("line.separator"));
			finalOutput.append("recall: ").append(recall).append(System.getProperty("line.separator"));
			finalOutput.append("F1: ").append(F1).append(System.getProperty("line.separator"));
			finalOutput.append("FB0Half: ").append(FB0Half).append(System.getProperty("line.separator"));
			finalOutput.append("FB2: ").append(FB2).append(System.getProperty("line.separator")).append(System.getProperty("line.separator"));
			
			
			
			finalOutput.append("***********************************").append(System.getProperty("line.separator"));
			finalOutput.append(outputContent);
			
			System.out.println(finalOutput);
			
			writeIntoFile(Main.OUTPUT_ENSEMBLE_FILE_NAME, finalOutput.toString());
		} catch(Exception e){
			throw new RuntimeException("Exception Occured in reading test file.", e);
		}  finally {
			try {
				bufferedReader.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}	
	
	public static Node generateDecisionTree(Map<String, List<Tuple>> trainingTuplesMap, Set<Integer> attributeSet, boolean isCalledFromRandomForest){
		Node N = new Node();
		//If tuples in D are all of the same class C then return N lebeled with C
		if(trainingTuplesMap.keySet().size() == 1){
			// Get the label
			String label = null;
			for(String C : trainingTuplesMap.keySet()){
				label = C;
				break;
			}
			N.label = label;
			return N;			
		}
		
		//if attribute list is empty then return N labeled with the majority class in D
		if(attributeSet == null || attributeSet.isEmpty()){
			N.label = getMajorityClass(trainingTuplesMap);
			return N;
		}
		
		// Apply Attribute_selection_method to find the best splitting attribute
		
		int splittingAttribute = getBestSplittingAttribute(trainingTuplesMap, attributeSet, isCalledFromRandomForest);
		
		//Label node N with splitting_criterion
		N.label = Integer.toString(splittingAttribute);
		
		//TODO : Check what discrete and multiway splits means here
		//if splitting attribute is discrete valued and multiway splits allowed then remove the splitting attribute from the attribute set
		attributeSet.remove(splittingAttribute);
		
		// for each outcome j of splitting criterion; partition the tuples and grow subtrees for each partition
		//Create a Map with key as distinct values of splitting attribute and value as Map<String, List<Tuple>> with tuples having this value of splitting attribute
		Map<Integer, Map<String, List<Tuple>>> Djs = new HashMap<Integer, Map<String, List<Tuple>>>();
		for(String C : trainingTuplesMap.keySet()){
			for(Tuple t : trainingTuplesMap.get(C)){
				if(!Djs.containsKey(t.getAttributeValues()[splittingAttribute])){
					Djs.put(t.getAttributeValues()[splittingAttribute], new HashMap<String, List<Tuple>>());
				}
				if(Djs.get(t.getAttributeValues()[splittingAttribute]).get(t.getLabel()) == null){
					Djs.get(t.getAttributeValues()[splittingAttribute]).put(t.getLabel(), new ArrayList<Tuple>());
				}
				
				Djs.get(t.getAttributeValues()[splittingAttribute]).get(t.getLabel()).add(t);
			}
		}
		
		//Let Dj be the set of data tuples in D satisfying outcome j // a partition
		for(Integer j : Djs.keySet()){
			Map<String, List<Tuple>> Dj = Djs.get(j);
			if(N.childNodes == null){
				N.childNodes = new HashMap<Integer, Node>();
			}
			
			//If Dj is empty then attach a leaf labeled with the majority class in D to node N;
			if(Dj == null || Dj.isEmpty()){
				
				Node leafNode = new Node();
				leafNode.parentNode = N;
				leafNode.label = getMajorityClass(trainingTuplesMap);
				N.childNodes.put(j, leafNode);
			} else {
				//attach the node returned by generate_decision_tree(Dj, attribute_list) to Node N
				Node childNode  = generateDecisionTree(Dj, attributeSet, isCalledFromRandomForest);
				childNode.parentNode = N;
				N.childNodes.put(j, childNode);
			}
		}
		
		return N;
	}
	
	/**
	 * Gets the lebel or class with majority 
	 * @param trainingTuplesMap
	 * @return
	 */
	public static String getMajorityClass(Map<String, List<Tuple>> trainingTuplesMap){
		String majorityClass = null;
		int majorityClassTupleCount = 0;
		for(String C : trainingTuplesMap.keySet()){
			if(majorityClass == null){
				majorityClass = C;
				majorityClassTupleCount = trainingTuplesMap.get(C).size();
			} else if(majorityClassTupleCount < trainingTuplesMap.get(C).size()){
				majorityClass = C;
				majorityClassTupleCount = trainingTuplesMap.get(C).size();
			}
		}
		return majorityClass;
	}
	/**
	 * Returns best splitting attribute using Information Gain Attribute Selection Measure 
	 * @param trainingTuplesMap
	 * @param attributeSet
	 * @return
	 */
	public static int getBestSplittingAttribute(Map<String, List<Tuple>> trainingTuplesMap, Set<Integer> attributeSet, boolean isCalledFromRandomForest){
		
		Set<Integer> tempAttributeSet = new HashSet<Integer>();
		List<Integer> attributeList = new ArrayList<Integer>(attributeSet);
		Random rand = new Random();
		
		if(isCalledFromRandomForest && attributeList.size() > Main.F){
			
			// For random forest use only random F attributes (where F is a constant and much less than the total attributes
			while((tempAttributeSet.size() < Main.F)){
				tempAttributeSet.add(attributeList.get(rand.nextInt(attributeList.size())));
			}
			
			attributeSet = tempAttributeSet;
		}
		Map<Integer, Double> Gain = new HashMap<Integer, Double>();
		for(Integer attribute : attributeSet){
			Gain.put(attribute, (infoForDataSet(trainingTuplesMap) - infoForAttribute(trainingTuplesMap, attribute)));
		}
		int attributeWithHighestGain = 0;
		Double highestGain = 0.0;
		
		//Find attribute with highest gain
		for(Integer attribute1 : Gain.keySet()){
			if(attributeWithHighestGain == 0){
				attributeWithHighestGain = attribute1;
				highestGain = Gain.get(attributeWithHighestGain);
			}
			for(Integer attribute2 : Gain.keySet()){
				if(Gain.get(attribute2) > highestGain){
					attributeWithHighestGain = attribute2;
					highestGain = Gain.get(attributeWithHighestGain);
				}
			}
		}
		
		return attributeWithHighestGain;
			
	}
	
//	public static <K ,V extends Number> K getKeyWithMaxValue(Map<K, V> map){
//		
//		K keyWithHighestValue = null;
//		V highestValue = null ;
//		
//		//Find attribute with highest gain
//		for(K key1 : map.keySet()){
//			if(keyWithHighestValue == null){
//				keyWithHighestValue = key1;
//				highestValue = map.get(keyWithHighestValue);
//			}
//			for(K key2 : map.keySet()){
//				if(map.get(key2).highestValue){
//					keyWithHighestValue = key2;
//					highestValue = map.get(keyWithHighestValue);
//				}
//			}
//		}
//		
//		return attributeWithHighestGain;
//		
//	}
	
	/**
	 * Get the entropy of D
	 * @param trainingTuplesMap
	 * @return
	 */
	public static Double infoForDataSet(Map<String, List<Tuple>> trainingTuplesMap){
		//m is the number of distinct labels this dataset has
		int m = trainingTuplesMap.keySet().size();
		int totalNumberOfTuples = 0;
		for(String label : trainingTuplesMap.keySet()){
			totalNumberOfTuples = totalNumberOfTuples + trainingTuplesMap.get(label).size();
		}
		Double entropyOfD = 0.0;
		for(String label : trainingTuplesMap.keySet()){
			// calculate probabilty for this label
			Double probabilityForCurrentLabel = new Double(trainingTuplesMap.get(label).size()) / (new Double(totalNumberOfTuples));
			entropyOfD = entropyOfD + (probabilityForCurrentLabel * Math.log(probabilityForCurrentLabel));
			
		}
		//Negate the calculated entropy
		entropyOfD = entropyOfD * (-1);
		return entropyOfD;
	}
	
	/**
	 * 
	 * @param trainingTuplesMap
	 * @param attribute
	 * @return
	 */
	public static Double infoForAttribute(Map<String, List<Tuple>> trainingTuplesMap, int A){
		//v is the number of distinct values of attribute A
		int v = 0;
		Set<Integer> distinctValues = new HashSet<Integer>();
		int totalNumberOfTuples = 0;
		Double expectedInfo = 0.0;
		for(String label : trainingTuplesMap.keySet()){
			for(Tuple t : trainingTuplesMap.get(label)){
				distinctValues.add(t.getAttributeValues()[A]);
				totalNumberOfTuples++;
			}
		}
		v = distinctValues.size();
		
		for(Integer a : distinctValues){
			int numberOfTuplesWithAttributeValue = 0;
			Map<String, List<Tuple>> Dj = new HashMap<String, List<Tuple>>();
			
			for(String label : trainingTuplesMap.keySet()){
				for(Tuple t : trainingTuplesMap.get(label)){
					if(t.getAttributeValues()[A] == A){
						numberOfTuplesWithAttributeValue++;
						//Add into the training tuple map which has key as the label
			    		if(Dj.get(t.getLabel()) == null){
			    			Dj.put(t.getLabel(), new ArrayList<Tuple>());
			    			Dj.get(t.getLabel()).add(t);
			    		} else {
			    			Dj.get(t.getLabel()).add(t);
			    		}
					}
				}
			}
			expectedInfo = expectedInfo + ((numberOfTuplesWithAttributeValue / totalNumberOfTuples ) * infoForDataSet(Dj));			
		}
		
		return expectedInfo;	
		
	}
	
	/**
	 * Writes the output contents in the text file
	 * @param fileName
	 * @param outputContent
	 */
	private static void writeIntoFile(String fileName, String outputContent){
		// Write the output in txt file
				File file = new File(Main.OUTOUT_LOCATION+"\\"+fileName);
				Writer fileWriter = null;
				BufferedWriter bufferedWriter = null;
				try{
					fileWriter = new FileWriter(file);
					bufferedWriter = new BufferedWriter(fileWriter);
					bufferedWriter.write(outputContent);
					
				} catch(Exception e){
					throw new RuntimeException("Exception Occured while writing into "+fileName, e);
				} finally{
					if (bufferedWriter != null && fileWriter != null) {
						try{
							bufferedWriter.close();
							fileWriter.close();
						} catch(Exception e){
							e.printStackTrace();
						}
					}
				}
	}
	
	private static String predictClassForTuple(Node decisionTree, Tuple instance){
		String predictedClass = null;
		Set<String> classLabels = new HashSet<String>();
		classLabels.add("+1");
		classLabels.add("-1");
		
		Node currentNode = decisionTree;
		
		Node prevNode = null;//Just for debugging
		String currentNodeLabel = currentNode.label;
		int attribute = 0;
		int attributeValue = 0;
		int level = 0; //Just for debugging
		
		//Node label will either be one of class labels (+1, -1) or an attribute number
		while(!classLabels.contains(currentNodeLabel)){
			level++;
			prevNode = currentNode;
			attribute = Integer.parseInt(currentNodeLabel);
//			System.out.println(" attribute = "+attribute);
			
			attributeValue = instance.getAttributeValues()[attribute];
//			System.out.println("attributeValue = "+attributeValue);
			
			currentNode = currentNode.childNodes.get(attributeValue);
			if(currentNode == null){
//				System.out.println("!!!!!!!!!!!!!!ERROR --- Node is null");
			}
			currentNodeLabel = currentNode.label;					
		}
		
		predictedClass = currentNodeLabel;		
		
		return predictedClass;
	}
}

class Node{
	String label;
	Node parentNode;
	//List<Node> childNodes;
	//Key is the edge value (attribute value) and Value is the child node
	Map<Integer, Node> childNodes;
}
/**
 * Each Line is a tuple or instance and have attribute number and value pairs
 * @author Shivani
 *
 */
class Tuple{
	String label;
	int[] attributeValues = new int[Main.NUMBER_OF_ATTRIBUTES_IN_DATASET + 1];
	Set<Integer> attributeSet = new HashSet<Integer>();
	
	
	public String getLabel() {
		return label;
	}


	public void setLabel(String label) {
		this.label = label;
	}


	public int[] getAttributeValues() {
		return attributeValues;
	}


	public void setAttributeValues(int[] attributeValues) {
		this.attributeValues = attributeValues;
	}


	public Set<Integer> getAttributeSet() {
		return attributeSet;
	}


	public void setAttributeSet(Set<Integer> attributeSet) {
		this.attributeSet = attributeSet;
	}


	public Tuple(String tupleLine){
		StringTokenizer tokenizer = new StringTokenizer(tupleLine," ");
		try{
			String label = tokenizer.nextToken();
			this.label = label;

			while (tokenizer.hasMoreTokens()) {
				String token = tokenizer.nextToken();
				String[] indexvalue = token.split(":");
				int index = Integer.parseInt(indexvalue[0]);
				int value = Integer.parseInt(indexvalue[1]);
				
				attributeValues[index] = value;
				attributeSet.add(index);
			} 
		} catch(Exception e){
			System.out.println("Exception occured while tokenizing the instance = "+tupleLine);
		}
		
		
		
	}
}
