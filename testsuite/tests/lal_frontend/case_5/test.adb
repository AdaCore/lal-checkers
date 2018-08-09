procedure Test is
   type Kind is (BinExpr, UnExpr, Lit);

   expr_kind : Kind;
   is_op : Boolean;
begin
   case expr_kind is
      when BinExpr | UnExpr =>
         is_op := True;
      when Lit =>
         is_op := False;
   end case;
end Ex1;
