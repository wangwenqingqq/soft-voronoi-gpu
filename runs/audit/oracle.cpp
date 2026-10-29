#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
int ed(const std::string& a, const std::string& b) {
 std::array<int,257> row{};
 std::iota(row.begin(),row.end(),0);
 for(size_t i=1;i<=a.size();++i){
  int diagonal=row[0]; row[0]=int(i);
  for(size_t j=1;j<=b.size();++j){
   int old=row[j];
   row[j]=std::min({row[j]+1,row[j-1]+1,diagonal+int(a[i-1]!=b[j-1])});
   diagonal=old;
  }
 }
 return row[b.size()];
}
std::vector<std::string> read(const char* p){std::ifstream f(p);if(!f)throw std::runtime_error("read error");std::vector<std::string> v;for(std::string s;std::getline(f,s);){if(!s.empty()&&s.back()=='\r')s.pop_back();if(s.size()>256)throw std::runtime_error("long string");v.push_back(s);}return v;}
int main(int argc,char**argv){if(argc!=4)return 2;auto db=read(argv[1]), qs=read(argv[2]);int qi=std::stoi(argv[3]);std::vector<std::pair<int,int>> ds;ds.reserve(db.size());for(size_t i=0;i<db.size();++i)ds.emplace_back(ed(qs.at(qi),db[i]),int(i));std::partial_sort(ds.begin(),ds.begin()+10,ds.end());for(int i=0;i<10;++i)std::cout<<ds[i].first<<' '<<ds[i].second<<'\n';}
