/**
Serial SVD using QR decomposition
Final matrices of SVD are stored as below -
U in MatrixXd u
variance in VectorXd singular
v in MatrixXd v
*/

#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <fstream>
#include <iostream>
// #include <omp.h>
#include <sys/time.h>

using namespace std;
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> RMatrixXd;

int main(int argc, char *argv[]) {
  struct timeval start, end;
  unsigned long rows;
  int cols;

  if (argv[1] == NULL || argv[2] == NULL) {
    cout << "Mention rows and columns as arguments" << endl;
    exit(0);
  }

  rows = atol(argv[1]);
  cols = atoi(argv[2]);

  unsigned long long position = 0;
  double *data = new double[rows * cols];

  ifstream fin(argv[3]);
  if (fin.is_open()) {
    while (!fin.eof() && position < (rows * cols)) {
      fin >> data[position];
      position++;
    }
  } else {
    cout << "Input file name should be included in arguments" << endl;
    exit(0);
  }

  RMatrixXd a = Map<RMatrixXd>(data, rows, cols);
  // cout << a << endl;
  MatrixXd res(cols, cols);
  MatrixXd q1(rows, cols);

  gettimeofday(&start, NULL);

  LLT<MatrixXd> lltOfA(a.transpose() * a);
  res = lltOfA.matrixL().transpose();
  q1 = a * res.inverse();

  JacobiSVD<MatrixXd> svd(res, ComputeThinU | ComputeThinV);

  MatrixXd v = svd.matrixV();               // Matrix V of SVD
  VectorXd singular = svd.singularValues(); // Vector storing sigma

  q1 = q1 * svd.matrixU(); // final matrix U

  gettimeofday(&end, NULL);

  ofstream foutU, foutSigma, foutV;
  foutU.open("serial_u.dat", ofstream::trunc | ofstream::out);
  foutU << q1 << endl;
  foutSigma.open("serial_sigma.dat", ofstream::trunc | ofstream::out);
  foutSigma << singular << endl;
  foutV.open("serial_v.dat", ofstream::trunc | ofstream::out);
  foutV << v << endl;

  delete[] data;
}
