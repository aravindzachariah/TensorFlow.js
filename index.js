const tf = require('@tensorflow/tfjs');
const iris = require('./training.json');
const irisTesting = require('./testing.json');

const trainingData = tf.tensor2d(iris.map(item=> [
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]
),[130,4])

const testingData = tf.tensor2d(irisTesting.map(item=> [
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]),[14,4])

const outputData = tf.tensor2d(iris.map(item => [
    item.species === 'setosa' ? 1 : 0,
    item.species === 'virginica' ? 1 : 0,
    item.species === 'versicolor' ? 1 : 0

]), [130,3])

const model = tf.sequential();

model.add(tf.layers.dense({
    inputShape: [4],
    activation: "sigmoid",
    units: 10
}))


model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
model.summary();

const optimizer = tf.train.adam();

model.compile({
  optimizer: optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

async function train_data(){
    for(let i=0;i<15;i++){
       const res = await model.fit(trainingData,
                   outputData,{epochs: 40});  
       console.log(res.history.loss[0]);          
    }
 }

 async function main() {
    let train = await train_data();
    model.predict(testingData).print();
  }

main();