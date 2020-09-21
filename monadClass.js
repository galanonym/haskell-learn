// Monads are containers of functions that you can execute *later*.
// Those functions can take some (blocking) time to run.
// Once we call .preform() this function will be executed
class Monad {
  constructor(fn) {
    this.fn = fn;
  }

  flatMap(fn) {
    return new Monad(() => {
      return fn(this.fn());
    });
  }

  preform() {
    return this.fn();
  }
}

const a = new Monad(() => {
  console.log('Begin calculation...');
  sleep(1);
  return 'XXyyZZ';
});

const b = a.flatMap((calculated) => {
  console.log('Uppercasing...');
  sleep(1);
  uppercased = calculated.toUpperCase();
  return uppercased;
});

const c = b.flatMap((uppercased) => {
  console.log('Outputting...');
  sleep(1);
  console.log(uppercased)
});

// Blocking node sleep function
function sleep(n) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, n * 1000);
}

// Later in runtime
c.preform();
