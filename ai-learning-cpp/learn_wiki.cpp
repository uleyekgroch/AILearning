#include "ai_learning/learning/distributional_semantics.hpp"
#include "ai_learning/learning/embedding_trainer.hpp"
#include <cmath>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace ai_learning::learning;

std::string fmt(long long n) {
    char b[32];
    if(n>=1000000000){snprintf(b,32,"%.1fB",n/1e9);return b;}
    if(n>=1000000){snprintf(b,32,"%.1fM",n/1e6);return b;}
    if(n>=1000){snprintf(b,32,"%.1fK",n/1e3);return b;}
    return std::to_string(n);
}
std::string fmtb(long long b) {
    char buf[32];
    if(b>=1073741824){snprintf(buf,32,"%.2f GB",b/1073741824.0);return buf;}
    if(b>=1048576){snprintf(buf,32,"%.1f MB",b/1048576.0);return buf;}
    return std::to_string(b)+" B";
}

std::string extract_text(const std::string& line) {
    auto pos = line.find("\x22text\x22: \x22");
    if (pos == std::string::npos) return "";
    pos += 9;
    std::string r; r.reserve(4096);
    bool esc = false;
    for (size_t i = pos; i < line.size(); ++i) {
        char c = line[i];
        if (esc) {
            if(c==110||c==116) r+=32; else r+=c;
            esc = false;
        } else if (c==92) { esc=true; }
        else if (c==34) break;
        else { if(c==10||c==13) r+=32; else r+=c; }
    }
    return r;
}

std::vector<std::string> list_files(const std::string& dir) {
    std::vector<std::string> out;
    std::string cmd = "dir /s /b \x22"+dir+"\\wiki_*\x22 2>nul";
    FILE* p = _popen(cmd.c_str(),"r");
    if(!p) return out;
    char buf[1024];
    while(fgets(buf,sizeof(buf),p)) {
        std::string s(buf);
        while(!s.empty()&&(s.back()==10||s.back()==13)) s.pop_back();
        if(!s.empty()) out.push_back(s);
    }
    _pclose(p);
    return out;
}

float cosine_sim(const std::vector<float>& a, const std::vector<float>& b) {
    float dot=0,na=0,nb=0;
    for(size_t i=0;i<a.size();++i){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
    float d=std::sqrt(na)*std::sqrt(nb);
    return d<1e-8f?0.0f:dot/d;
}

int main(int argc, char* argv[]) {
    using clock = std::chrono::steady_clock;
    std::string wiki_dir = R"(D:\mayAi\AILearning_v0527\parallel-learning\data\wiki_zh)";
    if(argc>1) wiki_dir=argv[1];

    std::cout<<"==================================================\n";
    std::cout<<"  AILearning - wiki_zh\n";
    std::cout<<"  PPMI + SGNS\n";
    std::cout<<"==================================================\n\n";

    auto files = list_files(wiki_dir);
    std::cout<<"[0] Files: "<<files.size()<<"\n\n";
    if(files.empty()){std::cerr<<"No files!\n";return 1;}

    // Phase 1: PPMI
    std::cout<<"====== Phase 1: PPMI ======\n";
    auto t1 = clock::now();

    DistributionalSemanticsConfig dsc;
    dsc.min_cpt_freq = 5;
    dsc.window_size = 8;
    dsc.max_dimensions = 300;
    dsc.ppmi_threshold = 0.3;
    dsc.max_cpts = 100000;
    dsc.use_causal_prior = false;
    DistributionalSemantics ds(dsc);

    long long total_chars=0, total_articles=0;
    int fdone=0, report=std::max(1,(int)files.size()/20);

    for(auto& fp: files) {
        std::ifstream fin(fp, std::ios::binary);
        if(!fin) continue;
        std::string line;
        while(std::getline(fin,line)) {
            if(line.empty()||line[0]!=123) continue;
            auto txt=extract_text(line);
            if(txt.empty()) continue;
            ds.learn_from_text(txt);
            total_chars+=(long long)txt.size();
            total_articles++;
        }
        fdone++;
        if(fdone%report==0) {
            auto st=ds.stats();
            std::cout<<"  ["<<100*fdone/(int)files.size()<<"%] "
                <<fdone<<"/"<<files.size()<<" files, "
                <<fmt(total_articles)<<" articles, "
                <<fmt(total_chars)<<" chars, "
                <<st.cpts_represented<<" concepts\n";
        }
    }
    auto t1e=clock::now();
    double sec1=std::chrono::duration<double>(t1e-t1).count();
    auto ss=ds.stats();

    std::cout<<"\n  Phase 1: "<<sec1<<"s\n";
    std::cout<<"  Articles: "<<fmt(total_articles)<<", Chars: "<<fmtb(total_chars)<<"\n";
    std::cout<<"  Concepts: "<<ss.cpts_represented<<"\n";
    std::cout<<"  Dimensions: "<<ss.total_dimensions<<"\n";
    std::cout<<"  Avg density: "<<std::fixed<<std::setprecision(1)<<ss.avg_vector_density<<"\n\n";

    // Phase 1.5: PPMI eval
    std::cout<<"====== Phase 1.5: PPMI Eval ======\n";
    auto all_cpts = ds.all_cpts();
    std::cout<<"  Total concepts: "<<all_cpts.size()<<"\n";

    std::cout<<"\n  First 30 multi-char concepts:\n  ";
    int shown=0;
    for(auto& c: all_cpts) {
        if(shown>=30) break;
        if(c.size()>=6) { std::cout<<c<<" "; shown++; }
    }
    std::cout<<"\n\n";

    std::cout<<"  Pairwise similarity:\n";
    int pairs=0;
    for(int i=0; i<(int)all_cpts.size()&&pairs<10; i+=2) {
        auto& pa=all_cpts[i], &pb=all_cpts[i+1];
        if(pa.size()<6||pb.size()<6) continue;
        auto sim=ds.similarity(pa,pb);
        std::cout<<"    "<<pa<<" <-> "<<pb<<": "
            <<std::fixed<<std::setprecision(4)<<sim.similarity<<"\n";
        pairs++;
    }

    std::cout<<"\n  Most similar:\n";
    int qn=0;
    for(auto& w: all_cpts) {
        if(qn>=8) break;
        if(w.size()<6) continue;
        auto top=ds.most_similar(w,5);
        if(top.empty()) continue;
        std::cout<<"    "<<w<<" -> ";
        for(int i=0;i<std::min(5,(int)top.size());++i) {
            if(i) std::cout<<", ";
            std::cout<<top[i].cpt_b<<"("<<std::fixed<<std::setprecision(3)<<top[i].similarity<<")";
        }
        std::cout<<"\n";
        qn++;
    }

    std::cout<<"\n  Clusters (top 5):\n";
    auto clusters=ds.discover_clusters(0.15);
    for(int i=0;i<std::min(5,(int)clusters.size());++i) {
        auto& cl=clusters[i];
        std::cout<<"    ["<<cl.label<<"] ("<<cl.members.size()<<" members, "
            <<std::fixed<<std::setprecision(3)<<cl.cohesion<<"): ";
        for(int j=0;j<std::min(8,(int)cl.members.size());++j) {
            if(j) std::cout<<", ";
            std::cout<<cl.members[j];
        }
        if(cl.members.size()>8) std::cout<<", ...";
        std::cout<<"\n";
    }

    // Phase 2: SGNS
    std::cout<<"\n====== Phase 2: SGNS ======\n";
    auto t2=clock::now();

    EmbeddingTrainerConfig ec;
    ec.embedding_dim=128;
    ec.epochs=3;
    ec.min_count=5;
    ec.neg_samples=5;
    ec.window_size=8;
    ec.learning_rate=0.025;
    ec.seed=42;
    ec.report_interval=50000;
    ec.max_vocab=100000;
    EmbeddingTrainer trainer(ec);

    trainer.import_vocabulary(all_cpts);

    long long fed=0, max_feed=200000;
    std::cout<<"  Feeding corpus...\n";
    for(auto& fp: files) {
        if(fed>=max_feed) break;
        std::ifstream fin(fp,std::ios::binary);
        if(!fin) continue;
        std::string line;
        while(std::getline(fin,line)&&fed<max_feed) {
            if(line.empty()||line[0]!=123) continue;
            auto txt=extract_text(line);
            if(txt.empty()) continue;
            trainer.add_text(txt);
            fed++;
            if(fed%50000==0) std::cout<<"    fed "<<fmt(fed)<<" articles\n";
        }
    }
    std::cout<<"  Corpus: "<<fmt(fed)<<" articles\n";
    std::cout<<"  Training...\n\n";

    trainer.set_progress_callback([](const TrainingProgress& p){
        std::cout<<"    E"<<p.epoch<<"/"<<p.total_epochs
            <<" pairs="<<fmt(p.pairs_processed)
            <<" loss="<<std::fixed<<std::setprecision(4)<<p.loss
            <<" lr="<<std::setprecision(5)<<p.current_lr
            <<" "<<std::setprecision(1)<<p.elapsed_seconds<<"s\n";
    });

    auto res=trainer.train();
    auto t2e=clock::now();
    double sec2=std::chrono::duration<double>(t2e-t2).count();

    std::cout<<"\n  Phase 2: "<<sec2<<"s\n";
    std::cout<<"  Vocab: "<<res.vocab_size<<"  Dim: "<<res.embedding_dim<<"\n";
    std::cout<<"  Pairs: "<<fmt(res.total_pairs)<<"\n";
    std::cout<<"  Loss: "<<std::fixed<<std::setprecision(6)<<res.final_loss<<"\n";
    std::cout<<"  CUDA: "<<(res.used_cuda?"yes":"no")<<"\n\n";

    if(trainer.is_trained()) {
        std::cout<<"====== Phase 3: Embedding Eval ======\n";
        std::cout<<"\n  Dense similarity:\n";
        int ecnt=0;
        for(int i=0;i<std::min(20,(int)all_cpts.size());i+=2) {
            auto& pa=all_cpts[i],&pb=all_cpts[i+1];
            if(pa.size()<6||pb.size()<6) continue;
            auto va=trainer.get_embedding(pa),vb=trainer.get_embedding(pb);
            if(!va||!vb){std::cout<<"    "<<pa<<" <-> "<<pb<<": (no vec)\n";continue;}
            float s=cosine_sim(*va,*vb);
            std::cout<<"    "<<pa<<" <-> "<<pb<<": "<<std::fixed<<std::setprecision(4)<<s<<"\n";
            ecnt++;
        }

        std::cout<<"\n  Dense most similar:\n";
        int q2=0;
        for(auto& w: all_cpts) {
            if(q2>=8) break;
            if(w.size()<6) continue;
            auto top=trainer.most_similar(w,5);
            if(top.empty()) continue;
            std::cout<<"    "<<w<<" -> ";
            for(int i=0;i<std::min(5,(int)top.size());++i) {
                if(i) std::cout<<", ";
                std::cout<<top[i].word<<"("<<std::setprecision(3)<<top[i].similarity<<")";
            }
            std::cout<<"\n";
            q2++;
        }

        trainer.save("wiki_embeddings.bin");
        std::cout<<"\n  Saved: wiki_embeddings.bin\n";
    }

    std::cout<<"\n==================================================\n";
    std::cout<<"  SUMMARY\n";
    std::cout<<"==================================================\n";
    std::cout<<"  Corpus: "<<fmt(total_articles)<<" articles, "<<fmtb(total_chars)<<"\n";
    std::cout<<"  PPMI concepts: "<<ss.cpts_represented<<"\n";
    std::cout<<"  PPMI dims: "<<ss.total_dimensions<<"\n";
    std::cout<<"  Embedding vocab: "<<res.vocab_size<<"\n";
    std::cout<<"  Embedding dim: "<<res.embedding_dim<<"\n";
    std::cout<<"  Total time: "<<std::fixed<<std::setprecision(1)<<sec1+sec2<<"s\n";
    std::cout<<"==================================================\n";
    return 0;
}
