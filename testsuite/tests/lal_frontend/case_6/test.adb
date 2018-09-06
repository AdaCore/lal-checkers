procedure Test is
   type Kind is (BinExpr, UnExpr, Lit);

   expr_kind : Kind;
begin
   case expr_kind is
      when others =>
         expr_kind := BinExpr;
   end case;
end Ex1;
